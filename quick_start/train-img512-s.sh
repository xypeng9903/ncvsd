DISTRIBUTED_ARGS="--standalone --nproc_per_node=8"

torchrun $DISTRIBUTED_ARGS train_edm2.py \
    --preset="presets/img512-s.json" \
    --outdir="training-runs/img512-s" \
    --net="../model_zoo/edm2/edm2-img512-s-2147483-0.130.pkl" \
    --data="../data/edm2/img512-sd.zip" \
    --batch-gpu=64 \
    --duration="64Mi" \
    --checkpoint="4Mi" \
    --snapshot="1Mi" \
    --ts="10,22,39"