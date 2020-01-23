
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
