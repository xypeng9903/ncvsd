export CUDA_VISIBLE_DEVICES=0

# ffhq256
torchrun --standalone --nproc_per_node=1 train_edm2.py \
    --net="../model_zoo/NCVSD/ffhq_10m.pt" \
    --outdir="training-runs/ffhq256" \
    --data="../data/train/edm/ffhq256.zip" \
    --preset="presets/ffhq256.json" \
    --cond=False \
    --batch-gpu=2 \
    --batch=512 \
    --duration="128Mi" \
    --checkpoint="8Mi" \
    --snapshot="8Mi" \
    --grad-checkpoint=False \
    --fp16=False \