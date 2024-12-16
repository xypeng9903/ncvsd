NUM_GPUS=$1
BATCH_GPU=$2


#-----------------------------------------------------------------
# ImageNet 64x64

# img64-s
# torchrun --standalone --nproc_per_node=$NUM_GPUS train_edm2.py \
#     --net="../model_zoo/edm2/edm2-img64-s-1073741-0.075.pkl" \
#     --outdir="training-runs/img64-s" \
#     --data="../data/train/edm2/img64.zip" \
#     --preset="presets/img64-s.json" \
#     --batch-gpu=$BATCH_GPU \
#     --batch=2048 \
#     --duration="128Mi" \
#     --checkpoint="8Mi" \
#     --snapshot="1Mi"

# img64-m
# torchrun --standalone --nproc_per_node=$NUM_GPUS train_edm2.py \
#     --outdir="training-runs/img64-m" \
#     --net="../model_zoo/edm2/edm2-img64-m-2147483-0.060.pkl" \
#     --preset="presets/img64-m.json" \
#     --data="../data/train/edm2/img64.zip" \
#     --batch-gpu=$BATCH_GPU \
#     --batch=2048 \
#     --duration="128Mi" \
#     --checkpoint="8Mi" \
#     --snapshot="1Mi"

# img64-l
# torchrun --standalone --nproc_per_node=$NUM_GPUS train_edm2.py \
#     --outdir="training-runs/img64-l" \
#     --net="../model_zoo/edm2/edm2-img64-l-1073741-0.040.pkl" \
#     --preset="presets/img64-l.json" \
#     --data="../data/train/edm2/img64.zip" \
#     --batch-gpu=$BATCH_GPU \
#     --batch=2048 \
#     --duration="128Mi" \
#     --checkpoint="1Mi" \
#     --snapshot="1Mi"

#-----------------------------------------------------------------
# ImageNet 512x512

# img512-xs
# torchrun --standalone --nproc_per_node=$NUM_GPUS train_edm2.py \
#     --net="../model_zoo/edm2/edm2-img512-xs-2147483-0.135.pkl" \
#     --outdir="training-runs/img512-xs" \
#     --data="../data/train/edm2/img512-sd.zip" \
#     --preset="presets/img512-xs.json" \
#     --batch-gpu=$BATCH_GPU \
#     --batch=2048 \
#     --duration="128Mi" \
#     --checkpoint="1Mi" \
#     --snapshot="1Mi"
