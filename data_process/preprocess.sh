DATA_DIR=/workspace/data_process/data

# Binarized data
fairseq-preprocess \
  --user-dir /workspace/fairseq-yp/src \
  --task translation \
  --bertdict \
  --source-lang source \
  --target-lang target \
  --srcdict ${DATA_DIR}/vocab.txt \
  --tgtdict ${DATA_DIR}/vocab.txt \
  --trainpref ${DATA_DIR}/data_step1/train  \
  --validpref ${DATA_DIR}/data_step1/valid \
  --testpref ${DATA_DIR}/data_step1/test \
  --destdir ${DATA_DIR}/bin_src_kb  \
  --workers 60

