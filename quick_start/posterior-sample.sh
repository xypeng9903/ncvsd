TASK=$1

export HF_HOME="../model_zoo/huggingface"
export HF_ENDPOINT="http://hf-mirror.com"

python posterior_sample.py pixel \
    --net="training-runs/ffhq256-xs/snapshot-0000655-0.100.pkl" \
    --data="../data/test/ffhq_val" \
    --preset="presets/task/$TASK.yaml" \
    --outdir="out/$TASK" \
    --batch=1