NUM_GPUS=$1
BATCH_GPU=$2

export HF_HOME="../model_zoo/huggingface"
export HF_ENDPOINT="http://hf-mirror.com"

torchrun --standalone --nproc_per_node=$NUM_GPUS train_edm2.py \
    --preset="presets/img512-m.json" \
    --outdir="training-runs/img512-m" \
    --net="../model_zoo/edm2/edm2-img512-m-2147483-0.100.pkl" \
    --data="../data/edm2/img512-sd.zip" \
    --batch-gpu=$BATCH_GPU \
    --duration="64Mi" \
    --checkpoint="4Mi" \
    --snapshot="1Mi" \
    --ts="10,22,39"