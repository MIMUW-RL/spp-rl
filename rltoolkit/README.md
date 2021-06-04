# SPP-RL
Library Implementing the SPP-RL approach.


Ubuntu 20.4 install notes

1. download mujoco200 linux https://www.roboti.us/index.html and put into .mujoco directory with licence
add following line to .bashrc
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/cyranka/.mujoco/mujoco200/bin
```
2. install mujoco-py requirements
```
sudo apt install cmake
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```
3. install patchelf
```
sudo add-apt-repository ppa:jamesh/snap-support
sudo apt-get update
sudo apt install patchelf
```

4. 
```
pip install -r requirements.txt
```
