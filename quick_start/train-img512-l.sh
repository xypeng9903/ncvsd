DISTRIBUTED_ARGS="--standalone --nproc_per_node=8"

torchrun $DISTRIBUTED_ARGS train_edm2.py \
    --preset="presets/img512-l.json" \
    --outdir="training-runs/img512-l" \
    --net="../model_zoo/edm2/edm2-img512-l-1879048-0.085.pkl" \
    --data="../data/edm2/img512-sd.zip" \
    --batch-gpu=8 \
    --duration="64Mi" \
    --checkpoint="4Mi" \
    --snapshot="1Mi" \
    --ts="10,22,39"