NUM_GPUS=$1
NET=$2

torchrun --standalone --nproc_per_node=$NUM_GPUs calculate_metrics.py gen \
        --net=$NET \
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs \
        --seed=123456789