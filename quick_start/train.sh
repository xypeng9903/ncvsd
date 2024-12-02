export CUDA_VISIBLE_DEVICES=0

# ffhq256
torchrun --standalone --nproc_per_node=1 train_edm2.py \
    --preset="presets/ffhq256.json" \
    --outdir="training-runs/ffhq256" \
    --net="../model_zoo/NCVSD/ffhq_10m.pt" \
    --data="../data/train/edm/ffhq256.zip" \
    --cond=False \
    --batch-gpu=2 \
    --batch=128 \
    --duration="16Mi" \
    --checkpoint="128Ki" \
    --snapshot="128Ki" \
    --grad-checkpoint=False \
    --fp16=False \
