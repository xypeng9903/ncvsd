NUM_GPUS=$1
BATCH_GPU=$2


torchrun --standalone --nproc_per_node=$NUM_GPUS train_edm2.py \
    --preset="presets/ffhq256-v1.json" \
    --outdir="training-runs/ffhq256-add-gan-loss-v1" \
    --ts="10,22,39" \
    --batch-gpu=$BATCH_GPU \
    --duration="16Mi" \
    --checkpoint="128Ki" \
    --snapshot="128Ki" \
    --net="../model_zoo/ffhq_10m.pt" \
    --data="../data/edm2/ffhq256.zip" \
    --cond=False \
    --batch=512