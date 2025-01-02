NUM_GPUS=$1
BATCH_GPU=$2

torchrun --standalone --nproc_per_node=$NUM_GPUS train_edm2.py \
    --preset="presets/ffhq256-xs.json" \
    --outdir="training-runs/ffhq256-xs" \
    --net="../model_zoo/edm2/edm2-ffhq256.pkl" \
    --data="../data/edm2/ffhq256.zip" \
    --cond=False \
    --batch-gpu=$BATCH_GPU \
    --batch=256 \
    --duration="4Mi" \
    --checkpoint="64Ki" \
    --snapshot="64Ki" \
    --ts="10,22,39"