#export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0
DATA_DIR=data_process/data/bin_src_kb
#RESTORE_MODEL=model_pretrain/checkpoint72.pt
RESTORE_MODEL=checkpoints3/checkpoint_best.pt

nohup fairseq-train ${DATA_DIR} \
    --user-dir model \
    --task matchgo \
    --arch transformer_kplug_base \
    --bertdict \
    --sku2vec-path data_process/data/data_step1 \
    --img2ids-path data_process/data/data_step1 \
    --kb-repeat-path data_process/data/data_step1 \
    --kb-distinct-path data_process/data/data_step1 \
    --img2vec-path data_process/data/image_fc_vectors \
    --reset-optimizer --reset-dataloader --reset-meters \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
    --lr 0.00005 --stop-min-lr 1e-09 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy_rl --label-smoothing 0.1 \
    --update-freq 8 --max-tokens 1536 \
    --ddp-backend=no_c10d --max-epoch 30 \
    --max-source-positions 512 --max-target-positions 512 \
    --truncate-source \
    --save-interval-updates 500 \
    --restore-file ${RESTORE_MODEL} > log.train &

