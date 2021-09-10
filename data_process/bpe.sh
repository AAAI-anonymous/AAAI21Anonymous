DATA_DIR=data

# Tokenize
for split in train dev test; do
  for lang in source_kb_val target kb_val_repeat kb_val_distinct; do
    python multiprocessing_bpe_encoder.py \
      --vocab-bpe ${DATA_DIR}/vocab.txt \
      --inputs  ${DATA_DIR}/${split}.${lang} \
      --outputs ${DATA_DIR}/bpe_src_kb/${split}.${lang} \
      --workers 60
  done
done

