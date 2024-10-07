# hello_pinocchino


## Getting started python

```bash
conda env create -f pinocchino_env.yml
```



## Create conda env (optional)

```bash
conda create --name pinocchino python=3.10
conda activate pinocchino
conda install -c conda-forge mujoco
conda install -c conda-forge pinocchio
conda env export > pinocchino_env.yml
```

## Getting started cpp
install pinocchio 
- `git clone https://github.com/stack-of-tasks/pinocchio`
- `git checkout v3.2.0`
- `mkdir build && cd build`
- build with `cmake -DBUILD_PYTHON_INTERFACE=OFF ..`
- `sudo make -j24 install`

install hpp-fcl
- `git clone https://github.com/humanoid-path-planner/hpp-fcl
- `git checkout v2.4.5`
- `mkdir build && cd build`
- build with `cmake -DBUILD_PYTHON_INTERFACE=OFF ..`
- `sudo make -j24 install`
