NUM_GPUS=$1
NET=$2
TS=$3

export HF_HOME="../model_zoo/huggingface"
export HF_ENDPOINT="http://hf-mirror.com"

torchrun --standalone --nproc_per_node=$NUM_GPUs calculate_metrics.py gen \
        --net=$NET \
        --ts=$TS \
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs \
        --seed=123456789