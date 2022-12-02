ARG BASE_IMAGE="nvcr.io/nvidia/cuda:11.7.1-cudnn8-devel-ubuntu18.04"
FROM ${BASE_IMAGE} AS deps

ENV PYTHONUNBUFFERED TRUE

USER root

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y software-properties-common lsb-release && add-apt-repository -y ppa:deadsnakes/ppa \
  && apt-get update \
  && apt-get install -y sudo git mc vim cmake wget curl bison flex libbison-dev libfl-dev libfl2 autopoint \
  mosquitto mosquitto-clients \
  autoconf automake build-essential cmake git-core libass-dev libfreetype6-dev libgnutls28-dev libsdl2-dev libtool libva-dev libvdpau-dev libvorbis-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev \
  meson ninja-build pkg-config texinfo wget yasm zlib1g-dev nasm libx264-dev libvpx-dev libx265-dev libnuma-dev libfdk-aac-dev libunistring-dev libmp3lame-dev libopus-dev libsdl-pango-dev libjpeg-dev libbz2-dev libflac-dev libssl-dev libsrtp2-dev libmpeg2-4-dev libgirepository1.0-dev python3-cairo-dev \
  python3.8 python3.8-dev python3.8-distutils python3-pip \
  libtesseract-dev tesseract-ocr \
  && rm -rf /var/lib/apt/lists/*

# Cmake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
RUN apt-get update && apt install kitware-archive-keyring && rm /etc/apt/trusted.gpg.d/kitware.gpg && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6AF7F09730B3F0A4 && apt update && apt install -y cmake

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2 && update-alternatives --set python3 /usr/bin/python3.8 && python3 -m pip install --upgrade pip && pip install --upgrade setuptools && pip install --upgrade distlib

RUN addgroup --gid 1000 tstreamer && useradd -rm -d /home/tstreamer -s /bin/bash -G sudo -u 1000 --gid 1000 tstreamer && mkdir -p /home/tstreamer/tmp && echo 'tstreamer:tstreamer' | chpasswd

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh && chown -R tstreamer /home/tstreamer

USER tstreamer
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV HISTFILE=/home/bash/.bash_history_tstreamer
ENV PATH=/home/tstreamer/.local/bin:$PATH
ENV LD_LIBRARY_PATH=/home/tstreamer/.local/lib:/home/tstreamer/.local/lib/x86_64-linux-gnu:/usr/local/cuda/lib64/:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH
ENV PKG_CONFIG_PATH=/home/tstreamer/.local/lib/pkgconfig:/home/tstreamer/.local/lib/x86_64-linux-gnu/pkgconfig:/home/tstreamer/gst-plugins-base/build/pkgconfig/:/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
#ENV PKG_CONFIG_LIBDIR=$PKG_CONFIG_LIBDIR:/home/tstreamer/.local/lib
ENV GI_TYPELIB_PATH=/home/tstreamer/.local/lib/x86_64-linux-gnu/girepository-1.0/
ENV XDG_DATA_DIRS=/home/tstreamer/.local/share/:/usr/share/:$XDG_DATA_DIRS

RUN chown -R tstreamer:tstreamer /home/tstreamer

COPY --chown=tstreamer requirements.txt /home/tstreamer/requirements.txt
RUN pip install -r /home/tstreamer/requirements.txt && pip uninstall -y enum34 && rm /home/tstreamer/requirements.txt

#vulkan
USER root
RUN wget -qO - http://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add - && wget -qO /etc/apt/sources.list.d/lunarg-vulkan-bionic.list http://packages.lunarg.com/vulkan/lunarg-vulkan-bionic.list && apt update && apt install -y vulkan-sdk && rm -rf /var/lib/apt/lists/*

#nvidia libs
USER root
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH
#ARG NVIDIA_DRIVER_VERSION="455.38"
ARG NVIDIA_DRIVER_VERSION="515.65.01"
ENV NVIDIA_DRIVER_VERSION=${NVIDIA_DRIVER_VERSION}
RUN echo $NVIDIA_DRIVER_VERSION
COPY gst-nvenc/libnvidia-encode.so.${NVIDIA_DRIVER_VERSION} /usr/local/cuda/lib64/stubs/libnvidia-encode.so.1
COPY gst-nvenc/libnvcuvid.so.${NVIDIA_DRIVER_VERSION} /usr/local/cuda/lib64/stubs/libnvcuvid.so.1
RUN ln -s /usr/local/cuda/lib64/stubs/libnvidia-encode.so.1 /usr/local/cuda/lib64/stubs/libnvidia-encode.so && ln -s /usr/local/cuda/lib64/stubs/libnvcuvid.so.1 /usr/local/cuda/lib64/stubs/libnvcuvid.so && ldconfig

FROM deps as ffmpeg

# ffmpeg
USER tstreamer
WORKDIR /home/tstreamer
RUN git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg_source/ --branch release/5.1 --single-branch
WORKDIR /home/tstreamer/ffmpeg_source
RUN PATH="$HOME/.local/bin:$PATH" PKG_CONFIG_PATH="/usr/local/lib/pkgconfig" ./configure \
  --prefix="$HOME/.local" \
  --pkg-config-flags="--static" \
  --extra-cflags="-I$HOME/.local/include" \
  --extra-ldflags="-L$HOME/.local/lib" \
  --extra-libs="-lpthread -lm" \
  --ld="g++" \
  --bindir="$HOME/.local/bin" \
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
  --enable-pic \
  --enable-shared \
  --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --nvccflags="-gencode arch=compute_61,code=sm_61" --nvcc=/usr/local/cuda/bin/nvcc \
  --enable-pic \
  && PATH="$HOME/.local/bin:$PATH" && make -j$(nproc) && make install

FROM deps as gstreamer
COPY --from=ffmpeg /home/tstreamer/.local /home/tstreamer/.local

USER tstreamer
WORKDIR /home/tstreamer
RUN git clone https://github.com/GStreamer/orc.git \
  && git clone https://github.com/GStreamer/gstreamer.git --single-branch \
  && git clone https://github.com/GStreamer/gst-plugins-base.git --single-branch \
  && git clone https://github.com/GStreamer/gst-plugins-good.git --single-branch \
  && git clone https://github.com/GStreamer/gst-plugins-bad.git --single-branch \
  && git clone https://github.com/GStreamer/gst-plugins-ugly.git --single-branch \
  && git clone https://github.com/GStreamer/gst-libav.git --single-branch \
  && git clone https://github.com/GStreamer/gst-python.git --single-branch \
  && git clone https://github.com/GNOME/pygobject

USER tstreamer
RUN for gs in `echo "orc gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-libav pygobject gst-python"`; do \
cd /home/tstreamer/$gs; \
rm -rf /home/tstreamer/$gs/build; \
mkdir -p /home/tstreamer/$gs/build; \
cd /home/tstreamer/$gs/build; \
/home/tstreamer/.local/bin/meson --prefix="$HOME/.local" -Dbuildtype=release; \
ninja; \
ninja install; \
done
USER  root
RUN ldconfig

FROM deps as gsopencv
COPY --from=gstreamer /home/tstreamer/.local /home/tstreamer/.local

# opencv
USER tstreamer
WORKDIR /home/tstreamer
RUN git clone https://github.com/opencv/opencv.git &&  mkdir -p /home/tstreamer/opencv/build
WORKDIR /home/tstreamer/opencv/build
RUN git checkout 4.5.1 && wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.1.zip && unzip opencv_contrib.zip
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D PYTHON_EXECUTABLE=$(which python3) \
-D BUILD_opencv_python3=ON \
-D BUILD_opencv_python2=OFF \
-D BUILD_SHARED_LIBS=OFF \
-D BUILD_PYTHON_SUPPORT=ON \
-D BUILD_NEW_PYTHON_SUPPORT=ON \
-D CMAKE_INSTALL_PREFIX="$HOME/.local" \
-D PYTHON3_EXECUTABLE=$(which python3) \
-D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
-D PYTHON_INCLUDE_DIR2=$(python3 -c "from os.path import dirname; from distutils.sysconfig import get_config_h_filename; print(dirname(get_config_h_filename()))") \
-D PYTHON_LIBRARY=$(python3 -c "from distutils.sysconfig import get_config_var;from os.path import dirname,join ; print(join(dirname(get_config_var('LIBPC')),get_config_var('LDLIBRARY')))") \
-D PYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") \
-D OPENCV_PYTHON3_INSTALL_PATH="/home/tstreamer/.local/lib/python3.8/site-packages" \
-D OPENCV_EXTRA_MODULES_PATH=/home/tstreamer/opencv/build/opencv_contrib-4.5.1/modules \
-D WITH_GSTREAMER=ON \
-D WITH_CUDA=ON \
-D WITH_FFMPEG=ON \
-D OPENCV_FFMPEG_SKIP_BUILD_CHECK=ON \
-D WITH_V4L=ON \
-D WITH_GTK=OFF \
-D WITH_OPENCL=OFF \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D CUDA_ARCH_BIN=61 \
-D BUILD_EXAMPLES=OFF .. \
&& make -j$(nproc) \
&& make install
USER root
RUN ldconfig

FROM deps as final
COPY --from=gsopencv /home/tstreamer/.local /home/tstreamer/.local

USER tstreamer
RUN pip uninstall -y pillow && CC="cc -mavx2" && pip install -U --force-reinstall pillow-simd
RUN python3 -c "from PIL import features; print(features.check_feature('libjpeg_turbo'))"

USER tstreamer
WORKDIR /home/tstreamer
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["ls"]
