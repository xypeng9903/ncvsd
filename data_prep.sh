# python dataset_tool.py convert \
#         --source=/data0/pxy/ILSVRC/Data/CLS-LOC/train \
#         --dest=/data0/pxy/code/data/train/edm/img512.zip \
#         --resolution=512x512 \
#         --transform=center-crop-dhariwal

python dataset_tool.py encode \
        --source=/data0/pxy/code/data/train/edm/img512.zip \
        --dest=/data0/pxy/code/data/train/edm/img512-sd.zip