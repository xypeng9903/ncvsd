TASK=$1

export HF_HOME="../model_zoo/huggingface"
export HF_ENDPOINT="http://hf-mirror.com"


python posterior_sample.py pixel \
    --net="training-runs/ffhq256-xs-gan-warmup-0/snapshot-0002097-0.050.pkl" \
    --data="../data/test/ffhq_val" \
    --preset="presets/task/$TASK.yaml" \
    --outdir="out/$TASK" \
    --batch=25

# for beta in 0.0001; do
#     python posterior_sample.py pixel \
#         --net="training-runs/ffhq256-xs-gan-warmup-0/snapshot-0002097-0.050.pkl" \
#         --data="../data/test/ffhq_val" \
#         --preset="presets/task/$TASK.yaml" \
#         --outdir="out/$TASK-ema-beta-$beta" \
#         --beta=$beta \
#         --batch=25
# done