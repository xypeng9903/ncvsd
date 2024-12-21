NUM_GPUS=$1
BATCH_GPU=$2


torchrun --standalone --nproc_per_node=$NUM_GPUS train_edm2.py \
    --preset="presets/img64-s.json" \
    --outdir="training-runs/dev-img64-s" \
    --net="../model_zoo/edm2/edm2-img64-s-1073741-0.075.pkl" \
    --data="../data/edm2/img64.zip" \
    --batch-gpu=$BATCH_GPU \
    --batch=2048 \
    --duration="256Mi" \
    --checkpoint="512Ki" \
    --snapshot="512Ki"