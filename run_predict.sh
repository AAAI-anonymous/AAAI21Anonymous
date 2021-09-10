export CUDA_VISIBLE_DEVICES=2
DATA_DIR=data_process/data/bin_src_kb
MODEL=checkpoints/checkpoint_1_2000.pt

nohup fairseq-generate $DATA_DIR \
    --path $MODEL \
    --user-dir model \
    --task matchgo \
    --bertdict \
    --sku2vec-path data_process/data/data_step1 \
    --img2ids-path data_process/data/data_step1 \
    --kb-repeat-path data_process/data/data_step1 \
    --kb-distinct-path data_process/data/data_step1 \
    --img2vec-path data_process/data/image_fc_vectors \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --batch-size 64 \
    --beam 5 \
    --min-len 50 \
    --truncate-source \
    --no-repeat-ngram-size 3 \
    2>&1 | tee output.txt & 
