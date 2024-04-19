set -o errexit

# pip install 'pydantic<2'
# pip install -r requirements/requirements.txt
# pip install -r requirements/requirements-wandb.txt # optional, if logging using WandB
# pip install -r requirements/requirements-tensorboard.txt # optional, if logging via tensorboard
# python ./megatron/fused_kernels/setup.py install # optional, if using fused kernels
# pip install debugpy debugpy-run

pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git@a651e2c24ecf97cbf367fd3f330df36760e1c597