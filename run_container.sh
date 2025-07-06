#! /bin/bash

SUPPORTED_BACKENDS=("tf" "pt")

arg_backend=""
# get named parameters
while [ $# -gt 0 ]; do
  case "$1" in
    -b=*)
      arg_backend="${1#*=}"
      ;;
    *)
      break
  esac
  shift
done

# check if the backend is supported
if ! [[ $(echo ${SUPPORTED_BACKENDS[@]} | fgrep -w $arg_backend) ]]
then
  echo "ERROR: Given AI backend is not supported!"
  exit 1
fi

# iterate over all positional arguments which represent the mounting volumes
mounted_volumes=""
while [ "$1" != "" ]; do
    mounted_volumes="${mounted_volumes} -v "$1""

    # Shift all the parameters down by one
    shift

done

# create a container and run `container_entry`` script
# required to run with `-t` switch for later interactive mode execution! 
container_id=$(docker run --rm -t --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --env-file .env ${mounted_volumes} -v $(pwd):/metron_ai_deepforge_repo \
  --user "$(id -u):$(id -g)" \
  --volume "${HOME}:/${HOME}:ro" \
  --name metron_ai_deepforge_${arg_backend} \
  --detach \
  metron_ai/deepforge_${arg_backend}:latest)
docker exec $container_id "/metron_ai_deepforge_repo/container_entry.sh"

# exec interactive container
docker exec -it $container_id bash
