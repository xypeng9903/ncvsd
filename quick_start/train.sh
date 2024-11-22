export CUDA_VISIBLE_DEVICES=2,3

torchrun --standalone --nproc_per_node=2 train_edm2.py \
    --outdir="training-runs/00000-edm2-img64-xs" \
    --data="../data/train/edm/edm2-imagenet-64x64.zip" \
    --preset="configs/img64-s.json" \
    --batch-gpu=32 \
    --batch=2048 \
    --duration="1024Mi"