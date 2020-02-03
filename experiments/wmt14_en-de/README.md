
## Baseline

This is an experiment designed to replicate the results of
[Vaswani et. al](https://arxiv.org/abs/1706.03762).

```
python ../../train.py \
    runs/baseline -s en -t de \
    --trainprefs data/train.en-de.tok --validprefs data/valid.en-de.tok \
    --dictpref data/dict.en-de --bpepref data/code.en-de \
    --num_workers 2 --share_all_embeddings \
    --dropout 0.3 --attention_dropout 0.1 --activation_dropout 0.1 \
    --optimizer adam --adam_betas '(0.9, 0.98)' \
    --weight_decay 0.0 --label_smoothing 0.1 \
    --warmup_updates 4000 --lr 0.128 --min_lr 1e-9 \
    --max_batch_tokens 8192 --max_update 100000 --update_freq 8 \
    --save_interval 5000 --valid_interval 1000 --log_interval 100
```
