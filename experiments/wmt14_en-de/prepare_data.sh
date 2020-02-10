#!/bin/bash
set -e
# Adapted from https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-wmt14en2de.sh
# which as adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

if [ -d ../mosesdecoder ]; then
    echo 'Skip downloading Moses github repository'
else
    echo 'Cloning Moses github repository (for tokenization scripts)...'
    git clone https://github.com/moses-smt/mosesdecoder.git ../mosesdecoder
fi

NORM_PUNC="sacremoses normalize"
TOKENIZER="sacremoses tokenize"
CLEAN="perl ../mosesdecoder/scripts/training/clean-corpus-n.perl"
LEARN_BPE="subword-nmt learn-bpe"
APPLY_BPE="subword-nmt apply-bpe"
LEARN_VOCAB="python ../../learn_vocab.py"

RAW_DIR="../raw"
TEMP_DIR="./tmp"
DATA_DIR="./data"

URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://statmt.org/wmt14/training-parallel-nc-v9.tgz"
    "http://statmt.org/wmt14/test-full.tgz"
)
FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-nc-v9.tgz"
    "test-full.tgz"
)
CORPORA=(
    "training/europarl-v7.de-en"
    "commoncrawl.de-en"
    "training/news-commentary-v9.de-en"
)

src=en
tgt=de
lang=$src-$tgt
num_bpe_tokens=32000
num_workers=8

mkdir -p $RAW_DIR $TEMP_DIR $DATA_DIR

echo 'Downloading files'
for ((i=0;i<${#URLS[@]};++i)); do
    url=${URLS[i]}
    file=$RAW_DIR/${FILES[i]}

    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        curl -o $file "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar -C $RAW_DIR -zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar -C $RAW_DIR -xvf $file
        fi
    fi
done

echo 'Extract testing data'
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $RAW_DIR/test-full/newstest2014-deen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" > $DATA_DIR/test.$lang.$l
    echo ''
done

echo 'Preprocessing training data'
for l in $src $tgt; do
    file=$TEMP_DIR/clean.$lang.tok.$l
    rm -f $file
    for f in "${CORPORA[@]}"; do
        cat $RAW_DIR/$f.$l | \
            sed -e 's/\r/ /g' | \
            $NORM_PUNC -j $num_workers -l $l \
                --normalize-quote-commas --normalize-numbers | \
            $TOKENIZER -j $num_workers -a -l $l >> \
            $file
    done
done

echo 'Filtering training data'
$CLEAN --ratio 1.5 \
    $TEMP_DIR/clean.$lang.tok $src $tgt \
    $TEMP_DIR/filter.$lang.tok 1 250

echo 'Splitting train and valid...'
for l in $src $tgt; do
    awk '{if (NR%100 == 0)  print $0; }' $TEMP_DIR/filter.$lang.tok.$l \
        > $DATA_DIR/valid.$lang.tok.$l
    awk '{if (NR%100 != 0)  print $0; }' $TEMP_DIR/filter.$lang.tok.$l \
        > $TEMP_DIR/train.$lang.tok.$l
    echo ''
done

echo 'Shuffle training data'
paste $TEMP_DIR/train.$lang.tok.$src $TEMP_DIR/train.$lang.tok.$tgt | shuf | \
    awk -F "\t" "{
        print \$1 > \"$DATA_DIR/train.$lang.tok.$src\";
        print \$2 > \"$DATA_DIR/train.$lang.tok.$tgt\"
    }"

echo 'Learn BPE'
cat $DATA_DIR/train.$lang.tok.$src $DATA_DIR/train.$lang.tok.$tgt | \
    $LEARN_BPE -s $num_bpe_tokens > $DATA_DIR/code.$lang.$src
cp $DATA_DIR/code.$lang.$src $DATA_DIR/code.$lang.$tgt

echo 'Learn vocab'
cat $DATA_DIR/train.$lang.tok.$src $DATA_DIR/train.$lang.tok.$tgt | \
    $APPLY_BPE -c $DATA_DIR/code.$lang.$src | \
    $LEARN_VOCAB --threshold 20 --nwords $num_bpe_tokens --show_pbar \
    > $DATA_DIR/dict.$lang.$src
cp $DATA_DIR/dict.$lang.$src $DATA_DIR/dict.$lang.$tgt
