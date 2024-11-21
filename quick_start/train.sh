export CUDA_VISIBLE_DEVICES=2,3

# Train XS-sized model for ImageNet-512 using 8 GPUs
torchrun --standalone --nproc_per_node=2 train_edm2.py \
    --outdir=training-runs/00000-edm2-img64-xs \
    --data=../data/train/edm/edm2-imagenet-64x64.zip \
    --preset=edm2-img64-s \
    --batch-gpu=32