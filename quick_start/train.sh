NUM_GPUS=$1
BATCH_GPU=$2

#---------------------------------------------------------
# batch size 2048

# img64-s
# torchrun --standalone --nproc_per_node=$NUM_GPUS train_edm2.py \
#     --net="../model_zoo/edm2/imagenet64-s.pkl" \
#     --outdir="training-runs/img64-s" \
#     --data="../data/train/edm/img64.zip" \
#     --preset="presets/img64-s.json" \
#     --batch-gpu=$BATCH_GPU \
#     --batch=2048 \
#     --duration="128Mi" \
#     --checkpoint="1Mi" \
#     --snapshot="1Mi"

# img64-xl
# torchrun --standalone --nproc_per_node=$NUM_GPUS train_edm2.py \
#     --net="../model_zoo/edm2/imagenet64-s.pkl" \
#     --outdir="training-runs/img64-s" \
#     --data="../data/train/edm/img64.zip" \
#     --preset="presets/img64-s.json" \
#     --batch-gpu=$BATCH_GPU \
#     --batch=2048 \
#     --duration="128Mi" \
#     --checkpoint="1Mi" \
#     --snapshot="1Mi"

# img512-xs
# torchrun --standalone --nproc_per_node=$NUM_GPUS train_edm2.py \
#     --net="../model_zoo/edm2/edm2-img512-xs-2147483-0.135.pkl" \
#     --outdir="training-runs/img512-xs" \
#     --data="../data/train/edm/img512-sd.zip" \
#     --preset="presets/img512-xs.json" \
#     --batch-gpu=$BATCH_GPU \
#     --batch=2048 \
#     --duration="128Mi" \
#     --checkpoint="1Mi" \
#     --snapshot="1Mi"

#---------------------------------------------------------
# batch size 128

# img64-s
# torchrun --standalone --nproc_per_node=$NUM_GPUS train_edm2.py \
#     --outdir="training-runs/img64-s-bsz-128" \
#     --preset="presets/img64-s-bsz-128.json" \
#     --data="../data/train/edm/img64.zip" \
#     --net="../model_zoo/edm2/edm2-img64-s-1073741-0.075.pkl" \
#     --batch-gpu=$BATCH_GPU \
#     --batch=128 \
#     --duration="16Mi" \
#     --checkpoint="1Mi" \
#     --snapshot="1Mi"

# img64-xl
# torchrun --standalone --nproc_per_node=$NUM_GPUS train_edm2.py \
#     --outdir="training-runs/img64-xl-bsz-128" \
#     --preset="presets/img64-xl-bsz-128.json" \
#     --data="../data/train/edm/img64.zip" \
#     --net="../model_zoo/edm2/edm2-img64-xl-0671088-0.040.pkl" \
#     --batch-gpu=$BATCH_GPU \
#     --batch=128 \
#     --duration="16Mi" \
#     --checkpoint="1Mi" \
#     --snapshot="1Mi"