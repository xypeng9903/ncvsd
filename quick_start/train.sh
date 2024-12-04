export CUDA_VISIBLE_DEVICES=6,7

# ffhq256
torchrun --standalone --nproc_per_node=2 train_edm2.py \
    --preset="presets/ffhq256.json" \
    --outdir="training-runs/ffhq256" \
    --net="../model_zoo/NCVSD/ffhq_10m.pt" \
    --data="../data/train/edm2/ffhq-256x256.zip" \
    --cond=False \
    --batch-gpu=8 \
    --batch=128 \
    --duration="16Mi" \
    --checkpoint="128Ki" \
    --snapshot="128Ki" \
    --grad-checkpoint=True \
    --fp16=False \
