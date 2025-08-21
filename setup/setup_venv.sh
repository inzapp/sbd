#!/bin/bash
set -e

VENV_BASE_PATH=~/venv
VENV_NAME=sbd

VENV_PATH=${VENV_BASE_PATH}/${VENV_NAME}
PYTHON_SHORT_VERSION=$(echo "$(python3 --version)" | awk '{print $2}' | cut -d. -f1,2)

if [ ! -d ${VENV_PATH} ]; then
    echo "1. cuda 11.8 + python3.8"
    echo "2. cuda 11.8 + python3.10"
    echo -n "input : "
    read ANSWER
    if [ ${ANSWER} == "1" ]; then
        REQUIREMENTS_TXT_PATH=requirements_cu118_py38.txt
    elif [ ${ANSWER} == "2" ]; then
        REQUIREMENTS_TXT_PATH=requirements_cu118_py310.txt
    else
        echo "invalid input"
        exit
    fi

    echo "start install to ${VENV_PATH}"
    sudo apt install -y python${PYTHON_SHORT_VERSION}-venv
    mkdir -p ${VENV_BASE_PATH}
    python${PYTHON_SHORT_VERSION} -m venv ${VENV_PATH}
    source ${VENV_PATH}/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r ${REQUIREMENTS_TXT_PATH}
    deactivate

    echo "venv setup success to ${VENV_PATH}"
else
    echo "venv already setup to ${VENV_PATH}"
fi

echo && echo "run \"source ${VENV_PATH}/bin/activate\" for activating venv"
