# tensorflow/tensorflow:2.4.0-gpu
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
apt update
apt-add-repository -y ppa:deadsnakes/ppa
apt install -y vim git python3.8 python3.8-distutils python3.8-venv libgl1-mesa-glx
mkdir -p ~/venv/
python3.8 -m venv ~/venv/tf
source ~/venv/tf/bin/activate
cd ~/sbd
python3 -m pip install --upgrade pip
python3 -m pip install -r ./setup/requirements_cuda_110.txt
deactivate
echo 'source ~/venv/tf/bin/activate' >> ~/.bashrc
echo && echo && echo 'setup success. please exit this container and rerun run_docker.sh for stable usage.'
