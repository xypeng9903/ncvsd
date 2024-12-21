NUM_GPUS=$1
NET=$2

torchrun --standalone --nproc_per_node=$NUM_GPUs calculate_metrics.py gen \
        --net=$NET \
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64-baseline.npz \
        --seed=123456789