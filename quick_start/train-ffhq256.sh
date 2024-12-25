NUM_GPUS=$1
BATCH_GPU=$2


torchrun --standalone --nproc_per_node=$NUM_GPUS train_edm2.py \
    --preset="presets/ffhq256.json" \
    --outdir="training-runs/ffhq256" \
    --ts="10,22,39" \
    --batch-gpu=$BATCH_GPU \
    --duration="4Mi" \
    --checkpoint="64Ki" \
    --snapshot="64Ki" \
    --net="../model_zoo/ffhq_10m.pt" \
    --data="../data/edm2/ffhq256.zip" \
    --cond=False \
    --batch=256