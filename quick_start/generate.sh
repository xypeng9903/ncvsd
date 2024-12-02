NET=$1
STEPS=$2

export CUDA_VISIBLE_DEVICES=1

# ffhq
python generate_images.py \
    --net=$NET \
    --outdir="ffhq256/out" \
    --steps $STEPS \