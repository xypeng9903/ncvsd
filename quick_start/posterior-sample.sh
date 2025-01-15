TASK=$1

export HF_HOME="../model_zoo/huggingface"
export HF_ENDPOINT="http://hf-mirror.com"


python posterior_sample.py pixel \
    --net="training-runs/ffhq256-xs-gan-warmup-0/snapshot-0002097-0.050.pkl" \
    --data="../data/test/ffhq_val" \
    --preset="presets/task/$TASK.yaml" \
    --outdir="out/$TASK" \
    --batch=25

for ema_decay in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
    python posterior_sample.py pixel \
        --net="training-runs/ffhq256-xs-gan-warmup-0/snapshot-0002097-0.050.pkl" \
        --data="../data/test/ffhq_val" \
        --preset="presets/task/$TASK.yaml" \
        --outdir="out/$TASK-ema-decay-$ema_decay" \
        --ema-decay=$ema_decay \
        --batch=25
done

for beta in 1e-7 5e-7; do
    python posterior_sample.py pixel \
        --net="training-runs/ffhq256-xs-gan-warmup-0/snapshot-0002097-0.050.pkl" \
        --data="../data/test/ffhq_val" \
        --preset="presets/task/$TASK.yaml" \
        --outdir="out/$TASK-beta-$beta" \
        --beta=$beta \
        --batch=25
done