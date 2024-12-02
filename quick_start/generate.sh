NET=$1
STEPS=$2

export CUDA_VISIBLE_DEVICES=1

# ffhq
python generate_images.py \
    --net=$NET \
    --outdir="out/ffhq256" \
    --steps $STEPS \