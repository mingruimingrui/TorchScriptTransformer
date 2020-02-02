from __future__ import absolute_import, unicode_literals

import torch
from torch import Tensor
from typing import Tuple, List, Optional
from torch_script_transformer.modules.transformer import TransformerModel
from torch_script_transformer.data.dictionary import Dictionary


class BeamGenerator(torch.nn.Module):
    """ Use a model to perform beam search on src_tokens

    NOTE: Due to current TorchScript limitations a number of features
          has to be implemented in a less than ideal manner.
          - inplace assignments are not possible.
          - reorder_xxx methods cannot be imported from shared util folder
            due to to torchscript value resolution not working as intended when
            torchscript module is not in global()
    """

    def __init__(
        self, model, tgt_dict, beam_size=1,
        max_len=None, max_len_a=1.4, max_len_b=4.0,
        len_penalty=1.0, unk_penalty=0.0,
        no_repeat_ngram_size=0,
        init_out_tokens_with_eos=False
    ):
        """
        Args:
            model (TransformerModel): An TransformerModel instance.
            tgt_dict (Dictionary): The target language Dictionary.
            max_len (int): The maximum sequence length of generated output
                not including bos/eos. (default: max model allowable)
            max_len_a/b (float): Maximum sequence length
                will be (src_len * a + b). (default: (1.4, 4))
            len_penalty (float): < 1.0 favors shorter,
                > 1.0 favors longer. (default: 1.0)
            unk_penalty (float): > 0 produces less unk,
                < 0 produces more unk. (default: 1.0)
            no_repeat_ngram_size: tokens will not be repeated this many
                times consecutively. 0 means unrestricted. (default: 0)
            init_out_tokens_with_eos: out_tokens will start with
                eos instead of bos. (default: False)
        """
        super().__init__()
        assert isinstance(model, TransformerModel)
        assert isinstance(tgt_dict, Dictionary)

        self.model = model
        # self.bos_idx = torch.LongTensor([tgt_dict.bos_index])[0]
        # self.pad_idx = torch.LongTensor([tgt_dict.pad_index])[0]
        # self.unk_idx = torch.LongTensor([tgt_dict.unk_index])[0]
        # self.eos_idx = torch.LongTensor([tgt_dict.eos_index])[0]
        self.register_buffer(
            'bos_idx',
            torch.LongTensor([tgt_dict.bos_index])[0]
        )
        self.register_buffer(
            'pad_idx',
            torch.LongTensor([tgt_dict.pad_index])[0]
        )
        self.register_buffer(
            'unk_idx',
            torch.LongTensor([tgt_dict.unk_index])[0]
        )
        self.register_buffer(
            'eos_idx',
            torch.LongTensor([tgt_dict.eos_index])[0]
        )
        self.beam_size = beam_size

        max_model_allowable_len = model.decoder.max_target_positions - 2
        if not max_len:
            max_len = max_model_allowable_len
        self.max_len = min(max_len, max_model_allowable_len)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b

        # len_penalty and unk_penalty has to be registered as tensors
        # len_penalty = torch.FloatTensor([len_penalty])[0]
        # self.register_buffer('len_penalty', len_penalty)
        self.len_penalty = len_penalty
        unk_penalty = torch.FloatTensor([unk_penalty])[0]
        self.register_buffer('unk_penalty', unk_penalty)
        # self.neg_inf = torch.FloatTensor([float('-inf')])[0]
        self.register_buffer('neg_inf', torch.FloatTensor([float('-inf')])[0])

        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.init_out_tokens_with_eos = init_out_tokens_with_eos

    def half(self):
        self.model.half()

    @torch.no_grad()
    def forward(self, src_tokens):
        # type: (Tensor) -> Tuple[Tensor, Tensor, Tensor]

        device = src_tokens.device
        beam_size = self.beam_size
        beam_size_sq = int(beam_size * beam_size)
        bsz, src_len = src_tokens.size()
        max_tgt_len = self.determine_max_tgt_len(src_len)

        # Forward encoder
        encoder_out, encoder_padding_mask = \
            self.model.forward_encoder(src_tokens)

        # Create placeholder for outputs
        out_tokens, out_scores = \
            self.create_output_buffers(bsz, max_tgt_len)
        out_tokens = out_tokens.to(device)
        out_scores = out_scores.to(device)

        # Generate first token
        encoder_padding_mask = torch.jit.annotate(Optional[Tensor], None)
        logits, _, incremental_state = self.model.forward_decoder(
            prev_output_tokens=out_tokens[:, :1],
            encoder_out=encoder_out,
            encoder_padding_mask=encoder_padding_mask,
            incremental_state=None,
            need_attn=False
        )
        cand_scores, cand_tokens = self.determine_cands(logits)

        # Perform reordering and insert first tokens
        sent_order = torch.arange(bsz, dtype=torch.long) \
            .view(-1, 1) \
            .repeat(1, beam_size) \
            .view(-1).to(device)
        encoder_out, encoder_padding_mask = self.reorder_encoder_outs(
            sent_order, encoder_out, encoder_padding_mask)
        incremental_state = \
            self.reorder_incremental_state(sent_order, incremental_state)
        out_tokens, out_scores = \
            self.reorder_output_buffer(sent_order, out_tokens, out_scores)

        out_tokens[:, 1] = cand_tokens.view(-1)
        out_scores[:, 0] = cand_scores.view(-1)
        out_sent_scores = out_scores[:, 0] / self.get_score_norm(1)
        fin_pos = out_tokens[:, 1].eq(self.eos_idx)

        for step_nb in range(1, max_tgt_len):
            # Early exit if all are done
            if fin_pos.all():
                break

            # Do decoding to get candidates
            logits, _, incremental_state = self.model.forward_decoder(
                prev_output_tokens=out_tokens[:, step_nb:step_nb + 1],
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                need_attn=False
            )
            cand_scores, cand_tokens = self.determine_cands(logits)
            cand_scores += out_scores[:, step_nb - 1].unsqueeze(-1)
            cand_sent_scores = cand_scores / self.get_score_norm(step_nb + 1)

            # Account for previously completed sentences
            cand_tokens[fin_pos] = self.eos_idx
            cand_scores[fin_pos] = self.neg_inf
            cand_sent_scores[fin_pos, 1:] = self.neg_inf
            cand_sent_scores[fin_pos, 0] = out_sent_scores[fin_pos]

            # [bsz * beam_size, beam_size] -> [bsz, beam_size ** 2]
            cand_tokens = cand_tokens.view(bsz, beam_size_sq)
            cand_scores = cand_scores.view(bsz, beam_size_sq)
            cand_sent_scores = cand_sent_scores.view(bsz, beam_size_sq)
            # print(cand_sent_scores)

            # Sort to get top candidates
            cand_sent_scores, top_cands = \
                cand_sent_scores.topk(beam_size, 1, largest=True)
            cand_sent_scores = cand_sent_scores.view(-1)
            cand_tokens = cand_tokens.gather(1, top_cands).view(-1)
            cand_scores = cand_scores.gather(1, top_cands).view(-1)
            new_order = top_cands.view(-1) / beam_size + sent_order * beam_size

            # Reorder and insert candidates
            encoder_out, encoder_padding_mask = self.reorder_encoder_outs(
                new_order, encoder_out, encoder_padding_mask)
            incremental_state = \
                self.reorder_incremental_state(new_order, incremental_state)
            out_tokens, out_scores = \
                self.reorder_output_buffer(new_order, out_tokens, out_scores)
            out_tokens[:, step_nb + 1] = cand_tokens
            out_scores[:, step_nb] = cand_scores
            out_sent_scores = cand_sent_scores

            fin_pos = out_tokens[:, step_nb + 1].eq(self.eos_idx)

        # Form into (batch_size, beam_size, tgt_len)
        out_tokens = out_tokens.view(bsz, beam_size, -1)
        out_scores = out_scores.view(bsz, beam_size, -1)
        out_sent_scores = out_sent_scores.view(bsz, beam_size)

        return out_tokens, out_scores, out_sent_scores

    def determine_max_tgt_len(self, src_len):
        # type: (int) -> int
        return min(
            int(src_len * self.max_len_a + self.max_len_b),
            self.max_len
        )

    def create_output_buffers(self, bsz, max_tgt_len):
        # type: (int, int) -> Tuple[Tensor, Tensor]
        out_tokens = torch.zeros(
            bsz, max_tgt_len + 1, dtype=torch.long).fill_(self.pad_idx)
        out_scores = torch.zeros(
            bsz, max_tgt_len, dtype=torch.float32).fill_(self.neg_inf)
        if self.init_out_tokens_with_eos:
            out_tokens[:, 0] = self.eos_idx
        else:
            out_tokens[:, 0] = self.bos_idx
        return out_tokens, out_scores

    def determine_cands(self, logits):
        # type: (Tensor) -> Tuple[Tensor, Tensor]
        logits = logits[:, -1].float()  # Only interested in last pred

        # Never predict pad
        logits[:, self.pad_idx] = self.neg_inf
        # Apply unk penalty
        logits[:, self.unk_idx] -= self.unk_penalty

        # Compute softmax and get top k
        lprobs = logits.log_softmax(dim=-1)
        cand_scores, cand_tokens = lprobs.topk(k=self.beam_size, largest=True)
        return cand_scores, cand_tokens

    def get_score_norm(self, tgt_len):
        # type: (int) -> float
        """ Sentence scores are computed using vanilla beam search scoring
        instead of the sentence scores suggested by Wu et. al
        https://arxiv.org/abs/1609.08144

        S(Y|X) = log(P(Y|X)) / lp(Y)
        lp(Y) = |Y| ** (len_penalty - 1.0)

        This function returns |Y| * lp(Y) = |Y| ** len_penalty
        """
        return tgt_len ** self.len_penalty

    def reorder_encoder_outs(
        self, new_order, encoder_out, encoder_padding_mask
    ):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        """ Perform non-inplace encoder outs reordering """
        encoder_out = encoder_out[:, new_order]
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask[new_order]
        return encoder_out, encoder_padding_mask

    def reorder_incremental_state(self, new_order, incremental_state):
        # type: (Tensor, List[Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]) -> List[Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]
        """ Perform non-inplace incremental state reordering """
        new_incremental_state = torch.jit.annotate(
            List[Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]], [])
        for saved_state in incremental_state:
            (
                (
                    self_attn_saved_key,
                    self_attn_saved_value
                ),
                (
                    encoder_attn_saved_key,
                    encoder_attn_saved_value
                )
            ) = saved_state
            new_incremental_state.append((
                (
                    self_attn_saved_key[new_order],
                    self_attn_saved_value[new_order]
                ),
                (
                    encoder_attn_saved_key[new_order],
                    encoder_attn_saved_value[new_order]
                )
            ))

        return new_incremental_state

    def reorder_output_buffer(self, new_order, out_tokens, out_scores):
        # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
        """ Perform non-inplace incremental state reordering """
        out_tokens = out_tokens[new_order]
        out_scores = out_scores[new_order]
        return out_tokens, out_scores
