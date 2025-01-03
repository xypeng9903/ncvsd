NUM_GPUS=$1
BATCH_GPU=$2

export HF_HOME="../model_zoo/huggingface"
export HF_ENDPOINT="http://hf-mirror.com"

torchrun --standalone --nproc_per_node=$NUM_GPUS train_edm2.py \
    --preset="presets/img512-s-uncond.json" \
    --outdir="training-runs/img512-s-uncond" \
    --net="../model_zoo/edm2/edm2-img512-s-uncond-2147483-0.070.pkl" \
    --data="../data/edm2/img512-sd.zip" \
    --cond=False \
    --batch-gpu=$BATCH_GPU \
    --batch=2048 \
    --duration="32Mi" \
    --checkpoint="512Ki" \
    --snapshot="512Ki" \
    --ts="10,22,39"