# Specifies base image and tag
FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-0
WORKDIR /root

# Installs additional packages
RUN pip install seaborn

# Add /root to python path
ENV PYTHONPATH "${PYTHONPATH}:/root"

# Copie the code to the docker image.
COPY train_rpn.py /root/train_rpn.py

RUN mkdir -p /root/data
COPY data/__init__.py /root/data/__init__.py
COPY data/input_pipeline.py /root/data/input_pipeline.py
COPY data/kitti_classes.py /root/data/kitti_classes.py
COPY data/test_images /root/data/test_images

COPY models /root/models
COPY utils /root/utils

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "train_rpn.py"]