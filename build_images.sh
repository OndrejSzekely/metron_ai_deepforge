#!/bin/sh

docker build -t metron_ai/deepforge_tf:latest -f Dockerfile.tensorflow .
docker build -t metron_ai/deepforge_pt:latest -f Dockerfile.pytorch .