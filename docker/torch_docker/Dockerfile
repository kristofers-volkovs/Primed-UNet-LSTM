FROM anibali/pytorch:1.7.0-cuda11.0-ubuntu20.04

# Installs extra dependencies to conda and pip
RUN conda install -y -c conda-forge scikit-learn pytorch matplotlib numpy \
&& pip install tqdm tensorboardx tensorflow torch-optimizer torchnet \
&& conda clean -ya
