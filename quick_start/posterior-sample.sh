TASK=$1

export HF_HOME="../model_zoo/huggingface"
export HF_ENDPOINT="http://hf-mirror.com"

python posterior_sample.py latent \
    --net="training-runs/img512-s/snapshot-0008388-0.050.pkl" \
    --data="../data/test/ffhq_val" \
    --preset="presets/task/$TASK.yaml" \
    --outdir="out/$TASK" \
    --batch=2