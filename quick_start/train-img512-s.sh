NUM_GPUS=$1
BATCH_GPU=$2


torchrun --standalone --nproc_per_node=$NUM_GPUS train_edm2.py \
    --preset="presets/img512-s.json" \
    --outdir="training-runs/dev-img512-s" \
    --net="../model_zoo/edm2/edm2-img512-s-2147483-0.130.pkl" \
    --data="../data/edm2/img512-sd.zip" \
    --batch-gpu=$BATCH_GPU \
    --batch=128 \
    --duration="16Mi" \
    --checkpoint="1Mi" \
    --snapshot="128Ki"