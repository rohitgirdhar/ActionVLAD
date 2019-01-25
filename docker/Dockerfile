# Set the base as the nvidia-cuda Docker
FROM nvidia/cuda:8.0-devel

# Create directory for all of the files to go into and cd into it
WORKDIR /app

# Apt-get all needed dependencies
RUN apt-get update
RUN apt-get install -y git wget make gcc python python-pip build-essential curl \
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
RUN pip install --upgrade pip
RUN pip install numpy PILLOW h5py matplotlib scipy tensorflow-gpu==1.1.0
RUN git config --global url.https://github.com/.insteadOf git://github.com/

# Set variable for libcudnn.so.5
WORKDIR /
ENV CUDNN_PATH="/libcudnn.so.5"

# Clone git repo
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

WORKDIR /app/ActionVLAD/
