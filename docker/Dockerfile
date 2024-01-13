FROM nvcr.io/nvidia/pytorch:23.10-py3
# 530 driver 12.2
USER root

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN addgroup --gid 1000 tstreamer && useradd -rm -d /home/tstreamer -s /bin/bash -G sudo -u 1000 --gid 1000 tstreamer && mkdir -p /home/tstreamer/tmp && echo 'tstreamer:tstreamer' | chpasswd

RUN apt-get update && apt-get install -y sudo
RUN apt-get install -y git wget curl mc gcc g++ vim ffmpeg libsm6 libxext6 make tesseract-ocr sqlite3 iputils-ping netcat apt-transport-https ca-certificates libfreetype6-dev
RUN python -m pip install --upgrade pip
RUN pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12==23.12.*
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
RUN pip install opencv-python-headless==4.8.0.74

RUN apt-get install -y libfreetype6-dev
RUN pip uninstall -y pillow
RUN pip install --no-cache-dir pillow

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh && chown -R tstreamer /home/tstreamer

USER tstreamer
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV HISTFILE=/home/bash/.bash_history_tstreamer
ENV PATH=/home/tstreamer/.local/bin:/app:$PATH

WORKDIR /app
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["ls"]
