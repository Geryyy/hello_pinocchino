# hello_pinocchino


## Getting started

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

