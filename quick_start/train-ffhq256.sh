NUM_GPUS=$1
BATCH_GPU=$2


torchrun --standalone --nproc_per_node=$NUM_GPUS train_edm2.py \
    --preset="presets/ffhq256.json" \
    --outdir="training-runs/ffhq256-ddpm-sigmas" \
    --batch-gpu=$BATCH_GPU \
    --grad-checkpoint=False \
    --duration="16Mi" \
    --checkpoint="128Ki" \
    --snapshot="128Ki" \
    --net="../model_zoo/ffhq_10m.pt" \
    --data="../data/edm2/ffhq256.zip" \
    --cond=False \
    --batch=128