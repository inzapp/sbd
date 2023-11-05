### ----- user define vars -----
CUDA_VERSION=cu118
CONTAINER_NAME=sbd
TRAIN_DATA_PATH=~/train_data
### ----- user define vars -----

TAG=latest
SBD_PATH=$(pwd)/../
REPOSITORY_NAME=inzapp/sbd-$CUDA_VERSION
AVAILABLE_CUDA_VERSIONS=("cu102" "cu112" "cu118")
if [[ ! " ${AVAILABLE_CUDA_VERSIONS[*]} " == *" $CUDA_VERSION "* ]]; then
    echo "$CUDA_VERSION is nvalid cuda version. available cuda versions : ${AVAILABLE_CUDA_VERSIONS[*]}"
    exit
fi

if [ "$(docker images | grep $REPOSITORY_NAME | awk '{print $1}')" == "$REPOSITORY_NAME" ]; then
    docker run -u $(id -u):$(id -g) -d --name $CONTAINER_NAME -it --rm --gpus all -v $TRAIN_DATA_PATH:/train_data -v $SBD_PATH:/sbd $REPOSITORY_NAME:$TAG /bin/bash
else
    echo "Docker image not found : $REPOSITORY_NAME:$TAG"
    echo "Pull docker image using 'docker pull $REPOSITORY_NAME:$TAG'"
    echo "or build docker image with command 'docker build --no-cache -t $REPOSITORY_NAME:$TAG -f Dockerfile.$CUDA_VERSION .'"
    exit
fi

docker exec -it $CONTAINER_NAME /bin/bash
