TF_VERSION=2.5.1
CONTAINER_NAME=sbd
TENSORFLOW=tensorflow/tensorflow:$TF_VERSION-gpu
SBD_PATH=$(pwd)/../
TRAIN_DATA_PATH=~/train_data

if [ "$(docker ps | grep $TENSORFLOW | awk '{print $2}')" != "$TENSORFLOW" ]; then
    docker run -d --name $CONTAINER_NAME -it --rm --gpus all -v $TRAIN_DATA_PATH:/train_data -v $SBD_PATH:/root/sbd $TENSORFLOW /bin/bash
fi
docker exec -it $CONTAINER_NAME /bin/bash
