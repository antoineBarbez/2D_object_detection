# Specifies base image and tag
FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-0
WORKDIR /root

# Add /root to python path
ENV PYTHONPATH "${PYTHONPATH}:/root"

# Copies the trainer code to the docker image.
COPY train.py /root/train.py
RUN mkdir -p /root/data
COPY data/__init__.py /root/data/__init__.py
COPY data/input_pipeline.py /root/data/input_pipeline.py
COPY data/kitti_classes.py /root/data/kitti_classes.py

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "train.py"]