
## Baseline

This is an experiment designed to replicate the results of attention is all you
need.

```
# Learn vocabs to build token embeddings in transformer

python ../../learn_vocab.py \
    -s en -t de \
    --dictpref data/dict.en-de \
    --trainprefs data/train.en-de.tok \
    --src_bpe_path data/code \
    --tgt_bpe_path data/code \
    --joined_dictionary \
    --thresholdsrc 20 --thresholdtgt 20 \
    --nwordssrc 32000 --nwordstgt 32000 \
    --num_workers 8
```

Attempt to replicate https://github.com/pytorch/fairseq/issues/346.

```
python ../../train.py \
    runs/baseline \
    -s en -t de \
    --dictpref data/dict.en-de \
    --trainprefs data/train.en-de.tok \
    --validprefs data/valid.en-de.tok \
    --num_workers 1 \
    --src_bpe_path data/code \
    --tgt_bpe_path data/code \
    --dropout 0.3 \
    --attention_dropout 0.1 \
    --activation_dropout 0.1 \
    --optimizer adam --adam_betas '(0.9, 0.98)' \
    --clip_norm 0.0 --weight_decay 0.0 \
    --warmup_updates 4000 --lr 0.001 --min_lr 1e-9 \
    --label_smoothing 0.1 \
    --max_batch_tokens 8192 \
    --max_update 200000 --update_freq 2 \
    --save_interval 10000 \
    --valid_interval 10000 \
    --log_interval 100
```
