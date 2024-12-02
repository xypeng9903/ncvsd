export CUDA_VISIBLE_DEVICES=0

# img64-s (bsz 128)
torchrun --standalone --nproc_per_node=1 train_edm2.py \
    --outdir="training-runs/img64-s-bsz-128" \
    --preset="presets/img64-s-bsz-128.json" \
    --data="../data/train/edm/img64.zip" \
    --net="../model_zoo/edm2/imagenet64-s.pkl" \
    --batch-gpu=8 \
    --batch=128 \
    --duration="16Mi" \
    --checkpoint="1Mi" \
    --snapshot="1Mi" \
    --grad-checkpoint=False

# img64-s
# torchrun --standalone --nproc_per_node=1 train_edm2.py \
#     --net="../model_zoo/edm2/imagenet64-s.pkl" \
#     --outdir="training-runs/img64-s" \
#     --data="../data/train/edm/img64.zip" \
#     --preset="presets/img64-s.json" \
#     --batch-gpu=8 \
#     --batch=2048 \
#     --duration="128Mi" \
#     --checkpoint="1Mi" \
#     --snapshot="1Mi" \
#     --grad-checkpoint=False

# img512-xs
# torchrun --standalone --nproc_per_node=1 train_edm2.py \
#     --net="../model_zoo/edm2/img512-xs-uncond.pkl" \
#     --outdir="training-runs/img512-xs" \
#     --data="../data/train/edm/img512-sd.zip" \
#     --preset="presets/img512-xs.json" \
#     --batch-gpu=8 \
#     --batch=2048 \
#     --duration="128Mi" \
#     --checkpoint="8Mi" \
#     --snapshot="8Mi" \
#     --grad-checkpoint=False