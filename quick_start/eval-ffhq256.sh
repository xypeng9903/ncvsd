NUM_GPUS=$1
NET=$2
TS=$3

torchrun --standalone --nproc_per_node=$NUM_GPUS calculate_metrics.py gen \
        --net=$NET \
        --ts=$TS \
        --ref="../data/edm2/ffhq256.pkl" \
        --seed=123456789