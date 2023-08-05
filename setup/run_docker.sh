### ----- user define vars -----
CUDA_VERSION=cu118
CONTAINER_NAME=sbd
TRAIN_DATA_PATH=~/train_data
### ----- user define vars -----

SBD_PATH=$(pwd)/../
REPOSITORY_NAME=inzapp/sbd
AVAILABLE_CUDA_VERSIONS=("cu102" "cu110" "cu112" "cu118")
if [[ ! " ${AVAILABLE_CUDA_VERSIONS[*]} " == *" $CUDA_VERSION "* ]]; then
    echo "$CUDA_VERSION is nvalid cuda version. available cuda versions : ${AVAILABLE_CUDA_VERSIONS[*]}"
    exit
fi

if [ "$(docker images | grep $REPOSITORY_NAME | grep $CUDA_VERSION | awk '{print $2}')" == "$CUDA_VERSION" ]; then
    docker run -u $(id -u):$(id -g) -d --name $CONTAINER_NAME -it --rm --gpus all -v $TRAIN_DATA_PATH:/train_data -v $SBD_PATH:/sbd $REPOSITORY_NAME:$CUDA_VERSION /bin/bash
else
    echo "Docker image not found : $REPOSITORY_NAME:$CUDA_VERSION"
    echo "Pull docker image using 'docker pull inzapp/sbd:$CUDA_VERSION'"
    echo "or build docker image with command 'docker build --no-cache -t $REPOSITORY_NAME:$CUDA_VERSION -f Dockerfile.$CUDA_VERSION .'"
    exit
fi

docker exec -it $CONTAINER_NAME /bin/bash
