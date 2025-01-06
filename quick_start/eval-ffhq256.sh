NUM_GPUS=$1
NET=$2

torchrun --standalone --nproc_per_node=$NUM_GPUS calculate_metrics.py gen \
        --net=$NET \
        --ref="../data/edm2/ffhq256.pkl" \
        --seed=123456789