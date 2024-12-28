python dataset_tool.py convert \
        --source="../data/imagenet/ILSVRC/Data/CLS-LOC/train" \
        --dest="../data/edm2/img64.zip" \
        --resolution=64x64 \
        --transform=center-crop-dhariwal

python dataset_tool.py convert \
        --source="../data/imagenet/ILSVRC/Data/CLS-LOC/train" \
        --dest="../data/edm2/img512.zip" \
        --resolution=512x512 \
        --transform=center-crop-dhariwal

python dataset_tool.py encode \
        --source="../data/edm2/img512.zip" \
        --dest="../data/edm2/img512-sd.zip"