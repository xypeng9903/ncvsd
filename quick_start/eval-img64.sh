NUM_GPUS=$1
NET=$2

torchrun --standalone --nproc_per_node=$NUM_GPUS calculate_metrics.py gen \
        --net=$NET \
        --ref="https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img64.pkl" \
        --seed=123456789