# Set the base as the nvidia-cuda Docker
FROM nvidia/cuda:8.0-devel

# Set the proxies needed to work inside the Intel network
# If outside of the network, comment out
ENV http_proxy http://proxy-chain.intel.com:911
ENV https_proxy https://proxy-chain.intel.com:912

# Create directory for all of the files to go into and cd into it
WORKDIR /app

# Apt-get all needed dependencies
RUN apt-get update
RUN apt-get install -y git wget make gcc python3 python3-pip build-essential curl \
		 cmake libreadline-dev git-core libqt4-dev libjpeg-dev \
		 libpng-dev ncurses-dev imagemagick libzmq3-dev gfortran \
		 unzip gnuplot gnuplot-x11 sudo vim libopencv-dev google-perftools \
		 libgoogle-perftools-dev ffmpeg
RUN apt-get install -y --no-install-recommends libhdf5-serial-dev liblmdb-dev
RUN echo "LD_PRELOAD=/usr/lib/libtcmalloc.so.4" | tee -a /etc/environment
ENV LD_PRELOAD "/usr/lib/libtcmalloc.so.4:$LD_PRELOAD"

# Install cuDNN and the dev files for cuDNN
WORKDIR /
COPY ./docker_files/cuDNN.deb /
RUN dpkg -i cuDNN.deb
COPY ./docker_files/dev-cuDNN.deb /
RUN dpkg -i dev-cuDNN.deb

# Install needed python packages
RUN pip3 install numpy PILLOW h5py matplotlib scipy tensorflow-gpu==1.1.0
RUN git config --global url.https://github.com/.insteadOf git://github.com/

# Explicitly clone and install OpenBLAS (needed for torch)
# WORKDIR /
# RUN git clone https://github.com/xianyi/OpenBLAS.git
# WORKDIR /OpenBLAS
# RUN make NO_AFFINITY=1 USE_OPENMP=1
# RUN make install

# # Now install torch
# WORKDIR /app/
# RUN git clone https://github.com/torch/distro.git --recursive
# WORKDIR /app/distro
# RUN CMAKE_LIBRARY_PATH=/opt/OpenBLAS/include:/opt/OpenBLAS/lib:$CMAKE_LIBRARY_PATH TORCH_NVCC_FLAGS="-D__CUDA_NO_HALF_OPERATORS__" ./install.sh -b
# WORKDIR /app/distro/
# WORKDIR /app/distro/install/bin

# # Install HDF5
# RUN ./luarocks install "https://raw.github.com/deepmind/torch-hdf5/master/hdf5-0-0.rockspec"
# # Have torch show up in PATH
# ENV PATH "/root/distro/install/bin:$PATH"

# Set variable for libcudnn for torch
WORKDIR /
ENV CUDNN_PATH="/libcudnn.so.5"

# Clone vogue (change this line for skinny-vogue, if desired)
RUN git clone -b master https://github.com/rohitgirdhar/ActionVLAD.git /app/ActionVLAD --recursive
WORKDIR /app/ActionVLAD/

#copy weights
COPY ./docker_files/models/kmeans-init/hmdb51/rgb_conv5_kmeans64.pkl /app/ActionVLAD/models/kmeans-init/hmdb51/
COPY ./docker_files/models/PreTrained/2-stream-pretrained/hmdb51/flow/split1.ckpt /app/ActionVLAD/models/PreTrained/2-stream-pretrained/hmdb51/flow/
COPY ./docker_files/models/PreTrained/2-stream-pretrained/hmdb51/flow/split2.ckpt /app/ActionVLAD/models/PreTrained/2-stream-pretrained/hmdb51/flow/
COPY ./docker_files/models/PreTrained/2-stream-pretrained/hmdb51/flow/split3.ckpt /app/ActionVLAD/models/PreTrained/2-stream-pretrained/hmdb51/flow/


COPY ./docker_files/models/PreTrained/ActionVLAD-pretrained/hmdb51/flow/split1.ckpt /app/ActionVLAD/models/PreTrained/ActionVLAD-pretrained/hmdb51/flow/
COPY ./docker_files/models/PreTrained/ActionVLAD-pretrained/hmdb51/flow/split2.ckpt /app/ActionVLAD/models/PreTrained/ActionVLAD-pretrained/hmdb51/flow/
COPY ./docker_files/models/PreTrained/ActionVLAD-pretrained/hmdb51/flow/split3.ckpt /app/ActionVLAD/models/PreTrained/ActionVLAD-pretrained/hmdb51/flow/

COPY ./docker_files/models/PreTrained/ActionVLAD-pretrained/hmdb51/rgb/split1.ckpt /app/ActionVLAD/models/PreTrained/ActionVLAD-pretrained/hmdb51/rgb/
COPY ./docker_files/models/PreTrained/ActionVLAD-pretrained/hmdb51/rgb/split2.ckpt /app/ActionVLAD/models/PreTrained/ActionVLAD-pretrained/hmdb51/rgb/
COPY ./docker_files/models/PreTrained/ActionVLAD-pretrained/hmdb51/rgb/split3.ckpt /app/ActionVLAD/models/PreTrained/ActionVLAD-pretrained/hmdb51/rgb/

COPY ./docker_files/models/PreTrained/imagenet-trained-CUHK/vgg_16_action_rgb_pretrain_uptoConv5.ckpt /app/ActionVLAD/models/PreTrained/imagenet-trained-CUHK/

#copy video to test
COPY ./docker_files/soccer10.mp4 /app/ActionVLAD/demo/
# Copy over this temp file needed for hourglass
#WORKDIR /app/vogue/
#COPY ./docker_files/temp.h5 /app/vogue/src/core/hourglass/preds/

# Copy over this file also needed for hourglass
COPY ./docker_files/libcudnn.so.5 /

# Remove the install files for cuDNN
WORKDIR /
RUN rm -f cuDNN.deb dev-cuDNN.deb

# Install eigen
WORKDIR /app/
RUN wget http://bitbucket.org/eigen/eigen/get/3.3.5.tar.gz
RUN tar -xvf 3.3.5.tar.gz
RUN mv eigen-eigen-b3f3d4950030/ eigen/
WORKDIR /app/eigen/build
RUN pwd
RUN cmake ..
RUN make && make install

WORKDIR /app/ActionVLAD/