#!/bin/sh

container=dash_caster
DOCKER_TAG="ragingcomputer/dashcast-docker-mqtt"

echo "Stopping $container"
docker stop $container
echo "Removing $container"
docker rm $container
echo "Running $container"
docker run -d \
      --name $container \
      --restart=always \
      --network="host" \
      -e DEFAULT_DASHBOARD_URL="http://192.168.10.22:5050/Landscape" \
      -e DEFAULT_DASHBOARD_URL_FORCE="False" \
      -e DISPLAY_NAME="Kitchen TV" \
      -e IGNORE_CEC="True" \
      -e MQTT_SERVER="192.168.10.22" \
      -e MQTT_USERNAME="xaser" \
      -e MQTT_PASSWORD="SnetBil8a" \
      $DOCKER_TAG