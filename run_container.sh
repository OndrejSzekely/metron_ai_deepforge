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

docker run --rm -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --env-file .env ${mounted_volumes} -v $(pwd):/metron_ai_deepforge_repo \
  --user "$(id -u):$(id -g)" \
  --volume "/etc/passwd:/etc/passwd:ro" \
  --volume "/etc/group:/etc/group:ro" \
  --volume "${HOME}:/${HOME}:ro" \
  --name metron_ai_deepforge_${arg_backend} metron_ai/deepforge_${arg_backend}:latest \