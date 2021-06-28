### 1. Docker (Recommended)
#### [A. Docker](https://docs.docker.com/engine/install/ubuntu/)
```angular2html
# Uninstall old versions
sudo apt-get remove docker docker-engine docker.io containerd runc

# Install using the repository
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io

sudo docker run hello-world

# unkown error
systemctl start docker

# If this error is occured.
Job for docker.service failed because the control process exited with error code. See "systemctl status docker.service" and "journalctl -xe" for details.

vi /var/log/message
systemctl start docker
```

#### [B. Nvidia-driver](https://codechacha.com/ko/install-nvidia-driver-ubuntu/)

```angular2html
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
apt-cache search nvidia | grep nvidia-driver-460
sudo apt-get install nvidia-driver-460
sudo reboot

# If it occurs the error, please delete all of file related nivida.
sudo apt --purge autoremove nvidia*
```

#### [C. Nvidia-docker](https://github.com/NVIDIA/nvidia-docker) 

```angular2html
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
  
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```
#### D. UniRef30 
```angular2html
wget http://gwdu111.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz
tar -xvzf UniRef30_2020_06_hhsuite.tar.gz
```

#### E. Error
```
# local user
sudo groupadd docker
sudo usermod -aG docker $USER

# Permission denied
sudo chmod 666 /var/run/docker.sock
```

#### F. Build
```
cd [your path]/worksheet/official/base/docker
docker build -t pharmcadd:1.0 .  

# you can check your images .
docker images 

# If you are finished the building the docker file.  
# you can make the conatiner 
nvidia-docker run -it --network=host -v /mnt/share/ssh:/root/.ssh -v [your data path]:/data pharmcadd:1.0

# when you are in container
pip list 
# you can check your pip list, this container has the our environment
# you don't have to install cuda, tensorflow ... etc  

# when you want to exit.
ctrl-d or exit 

# Outside the container, you can check your container. 
docker ps or docker ps -a 

# ex) 
CONTAINER ID   IMAGE           COMMAND       CREATED        STATUS       PORTS     NAMES
6b8306ee337b   pharmcadd:1.0   "/bin/bash"   1 hours ago    Up 1 hours             [your_con]

# when you want to enter the container 
docker exec -it [your_con] /bin/bash

# re-start the stopped container 
docker start [your_con] 

```