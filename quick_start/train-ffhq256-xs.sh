DISTRIBUTED_ARGS="--standalone --nproc_per_node=8"

torchrun $DISTRIBUTED_ARGS train_edm2.py \
    --preset="presets/ffhq256-xs.json" \
    --outdir="training-runs/ffhq256-xs" \
    --net="../model_zoo/edm2/edm2-ffhq256-xs.pkl" \
    --data="../data/edm2/ffhq256.zip" \
    --cond=False \
    --batch-gpu=16 \
    --duration="4Mi" \
    --checkpoint="256Ki" \
    --snapshot="64Ki" \
    --ts="0,14,39" \
    --ls=0.1
