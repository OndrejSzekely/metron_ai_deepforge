FROM nvcr.io/nvidia/tensorflow:22.11-tf2-py3
RUN pip install -f requirements.txt
CMD [ "bash" ]