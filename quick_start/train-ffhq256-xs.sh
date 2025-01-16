NUM_GPUS=$1
BATCH_GPU=$2

torchrun --standalone --nproc_per_node=$NUM_GPUS train_edm2.py \
    --preset="presets/ffhq256-xs.json" \
    --outdir="training-runs/ffhq256-xs" \
    --net="training-runs/edm2-ffhq256-xs/network-snapshot-0013631-0.050.pkl" \
    --data="../data/edm2/ffhq256.zip" \
    --cond=False \
    --batch-gpu=$BATCH_GPU \
    --duration="2Mi" \
    --checkpoint="32Ki" \
    --snapshot="32Ki" \
    --ts="0,14,39" \
    --ls=0.1