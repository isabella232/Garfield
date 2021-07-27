#!/usr/bin/env bash

set -e

echo "Starting deployment of Garfield demonstrator..."

compose_file=~/docker-compose.yaml

docker-compose -f ${compose_file} down
docker-compose -f ${compose_file} pull
docker-compose -f ${compose_file} up -d

echo "Deployment completed."
