DISTRIBUTED_ARGS="--standalone --nproc_per_node=8"

torchrun $DISTRIBUTED_ARGS train_edm2.py \
    --preset="presets/img64-m.json" \
    --outdir="training-runs/img64-m" \
    --net="../model_zoo/edm2/edm2-img64-m-2147483-0.060.pkl" \
    --data="../data/edm2/img64.zip" \
    --batch-gpu=32 \
    --duration="64Mi" \
    --checkpoint="4Mi" \
    --snapshot="1Mi" \
    --ts="10,22,39"