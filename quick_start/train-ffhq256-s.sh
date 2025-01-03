NUM_GPUS=$1
BATCH_GPU=$2

torchrun --standalone --nproc_per_node=$NUM_GPUS train_edm2.py \
    --preset="edm2-ffhq256-s" \
    --outdir="training-runs/edm2-ffhq256-s" \
    --data="../data/edm2/ffhq256.zip" \
    --batch-gpu=$BATCH_GPU \
    --cond=False \
    --checkpoint 128Ki \
    --snapshot 128Ki