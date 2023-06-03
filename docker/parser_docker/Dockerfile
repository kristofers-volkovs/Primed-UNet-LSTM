FROM ubuntu:20.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Create a working dir
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.3-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p ~/miniconda && \
    rm ~/miniconda.sh && \
    conda install -y python==3.8.3 && \
    conda install -y -c conda-forge xarray eccodes netcdf4 numpy xesmf && \
    conda clean -ya

RUN  pip install --upgrade pip && \
    pip install cfgrib cdsapi tqdm && \
    rm -rf ~/.cache/pip

# Creates config file for CDS api
ARG secret_code
RUN echo "url: https://cds.climate.copernicus.eu/api/v2" > $HOME/.cdsapirc
RUN echo "key: $secret_code" >> $HOME/.cdsapirc
