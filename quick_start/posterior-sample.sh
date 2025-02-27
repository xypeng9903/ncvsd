TASK=$1

# python posterior_sample.py pixel \
#     --net="training-runs/ffhq256-xs/snapshot-0002097-0.050.pkl" \
#     --data="../data/test/ffhq_val" \
#     --preset="presets/task/$TASK.yaml" \
#     --outdir="out/$TASK-runs-4" \
#     --batch=4 \
#     --runs=4

# for rho in 2 3 4 5 6 7; do
#     python posterior_sample.py pixel \
#         --net="training-runs/ffhq256-xs/snapshot-0002097-0.050.pkl" \
#         --data="../data/test/ffhq_val" \
#         --preset="presets/task/$TASK.yaml" \
#         --outdir="out/$TASK-ema-rho-$rho" \
#         --rho=$rho \
#         --batch=4
# done

for ema_sigma in 100; do
    for ema_decay in 0.85 0.8 0.75 0.7 0.65 0.6 0.55 0.5 0.4 0.2 0.0; do
        outdir="out/$TASK-ema-sigma-$ema_sigma-ema-decay-$ema_decay"
        if [ -d "$outdir" ]; then
            echo "$outdir exits. Skip."
            continue
        fi
        python posterior_sample.py pixel \
            --net="training-runs/ffhq256-xs/snapshot-0002097-0.050.pkl" \
            --data="../data/test/ffhq_val" \
            --preset="presets/task/$TASK.yaml" \
            --outdir="out/$TASK-ema-sigma-$ema_sigma-ema-decay-$ema_decay" \
            --ema-decay=$ema_decay \
            --ema-sigma=$ema_sigma \
            --batch=25
    done
done

# for beta in 2e-3 3e-3 4e-3 5e-3; do
#     python posterior_sample.py pixel \
#         --net="training-runs/ffhq256-xs/snapshot-0002097-0.050.pkl" \
#         --data="../data/test/ffhq_val" \
#         --preset="presets/task/$TASK.yaml" \
#         --outdir="out/$TASK-beta-$beta" \
#         --beta=$beta \
#         --batch=25
# done