# PILCO Learner

Implementation of PILCO algorithm in Python. NOTE!!! This branch uses JAX for GPU parallelization; 

## Training Video

[cart_pole.mp4](https://raw.githubusercontent.com/cryscan/pilco-learner/master/cart_pole.mp4)

## Steps to set up the conda enviornment and get the animation

1. `sudo apt install ffmpeg`
2. `conda create -n pilco38 python=3.8 -y`
3. `conda activate pilco38`
4. `pip install --upgrade pip`
5. `NOTE!!! This is for CPU version: pip install --upgrade jax jaxlib`
6. For GPU version: `pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
   1. NOTE here that you have to manually specify the CUDA version, e.g. cuda12_pip → for CUDA 12.x and cuda11_pip → for CUDA 11.x
7. `pip install matplotlib`
8. `python3 cart_pole.py`
9. `python3 cart_doublependulum.py`
10. `python3 unicycle_riding.py`
