NUM_GPUS=$1
BATCH_GPU=$2

torchrun --standalone --nproc_per_node=$NUM_GPUS train_edm2.py \
    --preset="presets/img64-l.json" \
    --outdir="training-runs/img64-l" \
    --net="../model_zoo/edm2/edm2-img64-l-1073741-0.040.pkl" \
    --data="../data/edm2/img64.zip" \
    --batch-gpu=$BATCH_GPU \
    --duration="64Mi" \
    --checkpoint="4Mi" \
    --snapshot="1Mi" \
    --ts="10,22,39"