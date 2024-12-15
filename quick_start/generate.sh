<<<<<<< HEAD
NET=$1
STEPS=$2

export CUDA_VISIBLE_DEVICES=1

# ffhq
python generate_images.py \
    --net=$NET \
    --outdir="out/ffhq256" \
    --steps $STEPS \
    --batch 1
=======
python generate_images.py \
    --net="/data0/pxy/code/ncvsd-edm2/training-runs/img64-s-bsz-128/snapshot-0000000-0.050.pkl" \
    --outdir="out/img64-s"
>>>>>>> origin/dev-tensorboard
