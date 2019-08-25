#!/bin/bash
set -x #echo on

# Pricing https://cloud.google.com/compute/all-pricing

# https://cloud.google.com/deep-learning-vm/docs/images
export IMAGE_FAMILY="tf-latest-gpu-experimental" # or for conda "pytorch-latest-gpu" # or "pytorch-latest-cpu" for non-GPU instances
# https://cloud.google.com/compute/docs/gpus/
export ZONE="us-central1-b" # "us-west2-c" # budget: "us-west1-b"
export INSTANCE_NAME="keras"
export INSTANCE_TYPE="n1-highmem-8" # budget: "n1-highmem-4"

# budget: 'type=nvidia-tesla-k80,count=1'
gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --accelerator="type=nvidia-tesla-t4,count=1" \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=200GB \
        --metadata="install-nvidia-driver=True" \
        --preemptible