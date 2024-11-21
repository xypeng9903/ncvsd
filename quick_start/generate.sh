export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/data0/pxy/code/model_zoo/huggingface"

# python generate_images.py --net=/data0/pxy/code/model_zoo/edm2/imagenet64-s.pkl --outdir=out
python generate_images.py --net=/data0/pxy/code/model_zoo/edm2/img512-xs-uncond.pkl --outdir=out