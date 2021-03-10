FROM nvcr.io/nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

ENV PYTHONUNBUFFERED TRUE

USER root

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update
RUN apt-get install -y sudo
RUN apt-get install -y git mc vim cmake wget curl
#gstreamer
RUN apt-get install -y libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-pulseaudio libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
RUN apt-get install -y libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavresample-dev libdc1394-22-dev libharfbuzz-dev freetype2-demos libtesseract-dev tesseract-ocr
#mosquito
RUN apt-get install -y mosquitto mosquitto-clients
#ffmpeg
RUN apt-get -y install autoconf automake build-essential cmake git-core libass-dev libfreetype6-dev libgnutls28-dev libsdl2-dev libtool libva-dev libvdpau-dev libvorbis-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev meson ninja-build pkg-config texinfo wget yasm zlib1g-dev nasm libx264-dev libvpx-dev libx265-dev libnuma-dev libfdk-aac-dev libunistring-dev libmp3lame-dev libopus-dev
#python3.8
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get install -y python3.8 python3.8-dev
RUN apt-get install -y python3-pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
RUN update-alternatives --set python3 /usr/bin/python3.8
RUN python3 -m pip install --upgrade pip

RUN addgroup --gid 1000 tstreamer
RUN useradd -rm -d /home/tstreamer -s /bin/bash -G sudo -u 1000 --gid 1000 tstreamer && mkdir -p /home/tstreamer/tmp
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN echo 'tstreamer:tstreamer' | chpasswd

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh && chown -R tstreamer /home/tstreamer

USER tstreamer

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV HISTFILE=/home/bash/.bash_history_tstreamer
ENV PATH=$PATH:/home/tstreamer/.local/bin

RUN chown -R tstreamer:tstreamer /home/tstreamer

COPY --chown=tstreamer requirements.txt /home/tstreamer/requirements.txt
RUN pip install -r /home/tstreamer/requirements.txt
RUN pip uninstall -y enum34
RUN rm /home/tstreamer/requirements.txt

# opencv
USER tstreamer
WORKDIR /home/tstreamer
RUN git clone https://github.com/opencv/opencv.git
RUN mkdir -p /home/tstreamer/opencv/build
WORKDIR /home/tstreamer/opencv/build
RUN git checkout 4.5.1
RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.1.zip
RUN unzip opencv_contrib.zip
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D PYTHON_EXECUTABLE=$(which python3) \
-D BUILD_opencv_python3=ON \
-D BUILD_opencv_python2=OFF \
-D BUILD_PYTHON_SUPPORT=ON \
-D BUILD_NEW_PYTHON_SUPPORT=ON \
-D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
-D PYTHON3_EXECUTABLE=$(which python3) \
-D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
-D PYTHON_INCLUDE_DIR2=$(python3 -c "from os.path import dirname; from distutils.sysconfig import get_config_h_filename; print(dirname(get_config_h_filename()))") \
-D PYTHON_LIBRARY=$(python3 -c "from distutils.sysconfig import get_config_var;from os.path import dirname,join ; print(join(dirname(get_config_var('LIBPC')),get_config_var('LDLIBRARY')))") \
-D PYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") \
-D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
-D OPENCV_EXTRA_MODULES_PATH=/home/tstreamer/opencv/build/opencv_contrib-4.5.1/modules \
-D WITH_GSTREAMER=ON \
-D WITH_CUDA=ON \
-D WITH_FFMPEG=ON \
-D WITH_V4L=ON \
-D WITH_GTK=OFF \
-D WITH_OPENCL=OFF \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D BUILD_EXAMPLES=ON ..
USER root
WORKDIR /home/tstreamer/opencv/build
RUN make -j$(nproc)
RUN make install
RUN ldconfig

#vulkan
USER root
RUN wget -qO - http://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
RUN wget -qO /etc/apt/sources.list.d/lunarg-vulkan-bionic.list http://packages.lunarg.com/vulkan/lunarg-vulkan-bionic.list
RUN apt update
RUN apt install -y vulkan-sdk

#nvenc
USER tstreamer
WORKDIR /home/tstreamer
RUN git clone git://anongit.freedesktop.org/gstreamer/gst-plugins-bad
WORKDIR /home/tstreamer/gst-plugins-bad
RUN git checkout $(gst-launch-1.0 --version | grep version | tr -s ' ' '\n' | tail -1)
COPY gst-nvenc/nvEncodeAPI.h /usr/local/cuda/include
COPY gst-nvenc/cuviddec.h /usr/local/cuda/include
COPY gst-nvenc/nvcuvid.h /usr/local/cuda/include
COPY gst-nvenc/libnvidia-encode.so /usr/local/cuda/lib64/stubs
COPY gst-nvenc/libnvcuvid.so /usr/local/cuda/lib64/stubs
WORKDIR /home/tstreamer/gst-plugins-bad
RUN ./autogen.sh --disable-gtk-doc --noconfigure
ENV NVENCODE_CFLAGS="-I/usr/local/cuda/include"
RUN ./configure --with-cuda-prefix="/usr/local/cuda"
WORKDIR /home/tstreamer/gst-plugins-bad/sys/nvenc
RUN make
WORKDIR /home/tstreamer/gst-plugins-bad/sys/nvdec
RUN make
USER root
RUN cp /home/tstreamer/gst-plugins-bad/sys/nvenc/.libs/libgstnvenc.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0/
RUN cp /home/tstreamer/gst-plugins-bad/sys/nvdec/.libs/libgstnvdec.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/:/usr/local/cuda/lib64/stubs:/usr/local/lib/gstreamer-1.0
RUN ldconfig

# ffmpeg
USER tstreamer
WORKDIR /home/tstreamer
RUN git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg/
ENV PKG_CONFIG_PATH="/usr/local/lib/pkgconfig"
WORKDIR /home/tstreamer/ffmpeg
RUN ./configure \
  --prefix="/usr/local" \
  --pkg-config-flags="--static" \
  --extra-cflags="-I$HOME/ffmpeg_build/include" \
  --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
  --extra-libs="-lpthread -lm" \
  --ld="g++" \
  --bindir="$HOME/bin" \
  --enable-gpl \
  --enable-gnutls \
  --enable-libass \
  --enable-libfdk-aac \
  --enable-libfreetype \
  --enable-libmp3lame \
  --enable-libopus \
  --enable-libvorbis \
  --enable-libvpx \
  --enable-libx264 \
  --enable-libx265 \
  --enable-nonfree \
  --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --nvccflags="-gencode arch=compute_52,code=sm_52" --nvcc=/usr/local/cuda/bin/nvcc
# --enable-libaom \
#  --enable-libsvtav1 \
#  --enable-libdav1d \
USER root
RUN make
RUN make install

#nvcodec
USER tstreamer
WORKDIR /home/tstreamer
RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
WORKDIR /home/tstreamer/nv-codec-headers
USER root
RUN make
RUN make install

USER tstreamer
WORKDIR /home/tstreamer

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["ls"]