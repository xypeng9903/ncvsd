TASK=$1

python posterior_sample.py \
    --net="training-runs/ai4s-resume/snapshot-0003604-0.050.pkl" \
    --data="../data/test/ffhq_val" \
    --preset="presets/task/$TASK.yaml" \
    --outdir="out/$TASK" \
    --ts="10,20,39"