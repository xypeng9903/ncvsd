TASK=$1

python posterior_sample.py pixel \
    --net="training-runs/ffhq256-xs/snapshot-0002097-0.050.pkl" \
    --data="../data/test/ffhq_val" \
    --preset="presets/task/$TASK.yaml" \
    --outdir="out/$TASK" \
    --batch=25