TRAIN_DATA_PATH=~/train_data

IMAGE=sbd:latest
CONTAINER_NAME=sbd
SBD_PATH=$(pwd)/../

if [ "$(docker images | grep $CONTAINER_NAME | awk '{print $1}')" != "$CONTAINER_NAME" ]; then
    echo 'SBD docker image not found.'
    echo 'Build docker image first with command "docker build --no-cache -t sbd -f Dockerfile.cuxxx ."'
    exit
fi

if [ "$(docker ps | grep $IMAGE | awk '{print $2}')" != "$IMAGE" ]; then
    docker run -u $(id -u):$(id -g) -d --name $CONTAINER_NAME -it --rm --gpus all -v $TRAIN_DATA_PATH:/train_data -v $SBD_PATH:/sbd $IMAGE /bin/bash
fi

docker exec -it $CONTAINER_NAME /bin/bash
