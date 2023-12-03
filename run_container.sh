#!/bin/sh

# iterate over all positional arguments which represent the mounting volumes
mounted_volumes=""
while [ "$1" != "" ]; do
    mounted_volumes="${mounted_volumes} -v "$1""

    # Shift all the parameters down by one
    shift

done

docker run --rm -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --env-file .env ${mounted_volumes} -v $(pwd):/metron_ai_deepforge_repo --name metron_ai_deepforge metron_ai/deepforge:latest