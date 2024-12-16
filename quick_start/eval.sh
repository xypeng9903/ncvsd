NET=$1

export CUDA_VISIBLE_DEVICES=1

# img64
torchrun --standalone --nproc_per_node=1 calculate_metrics.py gen \
        --net=$NET \
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64-baseline.npz \
        --seed=123456789