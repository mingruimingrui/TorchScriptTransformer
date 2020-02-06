from __future__ import absolute_import, unicode_literals, division

import torch
from torch import Tensor
from typing import Tuple, List, Optional
from torch_script_transformer.data.dictionary import Dictionary


class BeamGenerator(torch.nn.Module):
    """ Use a model to perform beam search on src_tokens

    NOTE: Due to current TorchScript limitations, a number of features
    has to be implemented in a less than ideal manner.

    Here's a list of caveats that are worked around
        - There are problems with inplace assign
          Listed here are some caveats that has to be worked around
          - inplace assignments is not torchscript compliant
          - some problems still exist within torchscript value resolution
            so reorder_xxx methods are defined here instead
          - sequential generation is intended to work purely
    """

    def __init__(
        self, model, tgt_dict, beam_size=1,
        max_len=None, max_len_a=1.4, max_len_b=4.0,
        len_penalty=1.0, unk_penalty=0.0,
        no_repeat_ngram_size=0, init_out_w_bos=False
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
            init_out_w_bos: out_tokens will start with
                bos instead of eos. (default: False)
        """
        super().__init__()
        assert isinstance(tgt_dict, Dictionary)

        self.model = model
        bos_idx = torch.LongTensor([tgt_dict.bos_index])[0]
        pad_idx = torch.LongTensor([tgt_dict.pad_index])[0]
        unk_idx = torch.LongTensor([tgt_dict.unk_index])[0]
        eos_idx = torch.LongTensor([tgt_dict.eos_index])[0]
        self.register_buffer('bos_idx', bos_idx)
        self.register_buffer('pad_idx', pad_idx)
        self.register_buffer('unk_idx', unk_idx)
        self.register_buffer('eos_idx', eos_idx)
        self.beam_size = beam_size

        max_model_allowable_len = model.decoder.max_target_positions - 2
        if not max_len:
            max_len = max_model_allowable_len
        self.max_len = min(max_len, max_model_allowable_len)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b

        # Penalty can be defined as device side tensors
        len_penalty = torch.FloatTensor([len_penalty])[0]
        unk_penalty = torch.FloatTensor([unk_penalty])[0]
        self.register_buffer('len_penalty', len_penalty)
        self.register_buffer('unk_penalty', unk_penalty)

        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.init_out_w_bos = init_out_w_bos

        neg_inf = torch.FloatTensor([float('-inf')])[0]
        neg_one = torch.LongTensor([-1])[0]
        self.register_buffer('neg_inf', neg_inf)
        self.register_buffer('neg_one', neg_one)

    def half(self):
        """ Beam generator is designed to work in fp32 for numerical
        Softmax in fp16 is generally not advisable """
        self.model.half()

    @torch.no_grad()
    def forward(self, src_tokens):
        # type: (Tensor) -> List[List[Tuple[Tensor, Tensor, Tensor]]]

        device = src_tokens.device
        beam_size = self.beam_size
        beam_size_sq = int(beam_size ** 2)
        bsz, src_len = src_tokens.size()
        max_tgt_len = self.determine_max_tgt_len(src_len)

        # Forward encoder
        encoder_padding_mask = torch
        encoder_out, encoder_padding_mask = \
            self.model.forward_encoder(src_tokens)

        # Create placeholder for outputs
        # NOTE: out_scores is the cummulative log probs of each token
        out_tokens, out_scores = \
            self.create_output_buffers(bsz, max_tgt_len)
        out_tokens = out_tokens.to(device)
        out_scores = out_scores.to(device)

        # Generate first token
        logits, _, incremental_state = self.model.forward_decoder(
            prev_output_tokens=out_tokens[:, :1],
            encoder_out=encoder_out,
            encoder_padding_mask=encoder_padding_mask,
            incremental_state=None,
            need_attn=False
        )
        cand_scores, cand_tokens = self.determine_cands(
            logits, ensure_eos=False, ensure_not_eos=True)

        # Sentence order is used to record the position of each sentence in
        # batch. Later on we will retire sentences which are completed
        # So we need a way to identify which sentences are remaining as well
        # as the order they are arranged in the tensor.
        # Right now, there are not reordering so the order is just a range.
        sent_order = torch.arange(bsz, dtype=torch.long)
        num_sent_remaining = bsz

        # Repeat encoder outputs and output placeholder by beam size
        # and insert topk candidates
        # NOTE: tiled_range is a useful variable to have around
        # It is essentially range(bsz) tiled by beam_size
        # eg. given bsz = 3, and beam_size = 2
        # tiled_range = [0, 0, 1, 1, 2, 2]
        tiled_range = sent_order \
            .view(-1, 1) \
            .repeat(1, beam_size) \
            .view(-1).to(device)
        encoder_out, encoder_padding_mask = self.reorder_encoder_outs(
            tiled_range, encoder_out, encoder_padding_mask)
        incremental_state = self.reorder_incremental_state(
            tiled_range, incremental_state)
        out_tokens = out_tokens[tiled_range]
        out_scores = out_scores[tiled_range]
        out_tokens[:, 1] = cand_tokens.view(-1)
        out_scores[:, 0] = cand_scores.view(-1)
        # out_sent_scores = out_scores[:, 0] / self.get_score_norm(1)

        # Create placeholder for completed sentences
        finalized = torch.jit.annotate(List[List[Tuple[Tensor, Tensor, Tensor]]], [])
        for __ in range(bsz):
            hypots = torch.jit.annotate(List[Tuple[Tensor, Tensor, Tensor]], [])
            finalized.append(hypots)

        for step_nb in range(1, max_tgt_len + 1):
            # Do decoding to get candidates
            logits, _, incremental_state = self.model.forward_decoder(
                prev_output_tokens=out_tokens[:, step_nb:step_nb + 1],
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                need_attn=False
            )
            cand_scores, cand_tokens = self.determine_cands(
                logits=logits,
                ensure_eos=step_nb == max_tgt_len,
                ensure_not_eos=False
            )
            cand_scores += out_scores[:, step_nb - 1].unsqueeze(-1)
            cand_sent_scores = cand_scores / self.get_score_norm(step_nb + 1)

            # [bsz * beam_size, beam_size] ->[bsz, beam_size ** 2]
            cand_tokens = cand_tokens.view(-1, beam_size_sq)
            cand_scores = cand_scores.view(-1, beam_size_sq)
            cand_sent_scores = cand_sent_scores.view(-1, beam_size_sq)

            # Sort to get top [beam_size * 2] candidates
            # in case half of them are eos
            cand_sent_scores, top_cands = \
                cand_sent_scores.topk(beam_size * 2, 1, largest=True)
            cand_sent_scores = cand_sent_scores
            cand_tokens = cand_tokens.gather(1, top_cands)
            cand_scores = cand_scores.gather(1, top_cands)

            # Identify finished candidates (hypots)
            eos_pos = cand_tokens.eq(self.eos_idx)
            fin_pos = eos_pos[:, :beam_size]
            has_finished_cands = fin_pos.any()
            has_finished_sents = False
            if has_finished_cands:
                fin_is, fin_js = torch.where(fin_pos)
                for i, j in zip(fin_is, fin_js):
                    sent_idx = sent_order[i]
                    out_idx = top_cands[i, j] / beam_size + i * beam_size
                    hypot_tokens = out_tokens[out_idx, :step_nb + 2].clone()
                    hypot_tokens[-1] = self.eos_idx
                    hypot_scores = out_scores[out_idx, :step_nb + 1].clone()
                    hypot_scores[-1] = cand_scores[i, j]

                    if len(finalized[sent_idx]) < beam_size:
                        finalized[sent_idx].append((
                            hypot_tokens,
                            hypot_scores,
                            cand_sent_scores[i, j]
                        ))
                        if len(finalized[sent_idx]) >= beam_size:
                            sent_order[i] = self.neg_one
                            num_sent_remaining -= 1
                            has_finished_sents = True

            # Terminate if all sentences are finalized
            if num_sent_remaining == 0:
                break

            # Identify top [beam_size] candidates
            if has_finished_cands:
                # Re-sort to get top [beam_size] candidates
                cand_sent_scores.masked_fill(eos_pos, self.neg_inf)
                cand_sent_scores, top_cands_ = \
                    cand_sent_scores.topk(beam_size, 1, largest=True)
                top_cands = top_cands.gather(1, top_cands_)
                cand_tokens = cand_tokens.gather(1, top_cands_)
                cand_scores = cand_scores.gather(1, top_cands_)

            else:
                # Slice to get top [beam_size] candidates
                top_cands = top_cands[:, :beam_size]
                cand_tokens = cand_tokens[:, :beam_size]
                cand_scores = cand_scores[:, :beam_size]
                cand_sent_scores = cand_sent_scores[:, :beam_size]

            # Reorder based on remaining candidates
            new_order = top_cands.reshape(-1) / beam_size \
                + tiled_range[:out_tokens.size(0)] * beam_size
            if has_finished_sents:
                unfin_sent_pos = ~sent_order.eq(self.neg_one)
                sent_order = sent_order[unfin_sent_pos]
                new_order = new_order.view(
                    -1, beam_size)[unfin_sent_pos].view(-1)
                cand_tokens = cand_tokens[unfin_sent_pos]
                cand_scores = cand_scores[unfin_sent_pos]

            encoder_out, encoder_padding_mask = self.reorder_encoder_outs(
                new_order, encoder_out, encoder_padding_mask)
            incremental_state = self.reorder_incremental_state(
                new_order, incremental_state)
            out_tokens = out_tokens[new_order]
            out_scores = out_scores[new_order]

            out_tokens[:, step_nb + 1] = cand_tokens.reshape(-1)
            out_scores[:, step_nb] = cand_scores.reshape(-1)

        return finalized

    def determine_max_tgt_len(self, src_len):
        # type: (int) -> int
        return min(
            self.max_len,
            int(src_len * self.max_len_a + self.max_len_b)
        )

    def create_output_buffers(self, bsz, max_tgt_len):
        # type: (int, int) -> Tuple[Tensor, Tensor]
        out_tokens = torch.zeros(
            bsz, max_tgt_len + 2, dtype=torch.long).fill_(self.pad_idx)
        out_scores = torch.zeros(
            bsz, max_tgt_len + 1, dtype=torch.float32).fill_(self.neg_inf)
        if self.init_out_w_bos:
            out_tokens[:, 0] = self.bos_idx
        else:
            out_tokens[:, 0] = self.eos_idx

        return out_tokens, out_scores

    def determine_cands(self, logits, ensure_eos, ensure_not_eos):
        # type: (Tensor, bool, bool) -> Tuple[Tensor, Tensor]
        logits = logits[:, -1].float()  # Only interested in last pred
        lprobs = logits.log_softmax(dim=-1)

        # Never predict pad
        lprobs[:, self.pad_idx] = self.neg_inf

        # Apply unkown token penalty
        lprobs[:, self.unk_idx] -= self.unk_penalty

        if ensure_eos:
            # Always predict eos
            lprobs[:, :self.eos_idx] = self.neg_inf
            lprobs[:, self.eos_idx + 1:] = self.neg_inf

        if ensure_not_eos:
            # Don't predict eos
            lprobs[:, self.eos_idx] = self.neg_inf

        # Get top k
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
        return (tgt_len + 1) ** self.len_penalty

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
