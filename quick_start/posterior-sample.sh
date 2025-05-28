TASK=$1

python posterior_sample.py pixel \
    --net="../model_zoo/ncvsd/ncvsd-ffhq256-xs.pkl" \
    --data="../data/test/ffhq_val" \
    --preset="presets/task/$TASK.yaml" \
    --outdir="out/$TASK" \
    --batch=25