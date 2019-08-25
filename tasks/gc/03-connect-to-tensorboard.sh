#!/bin/bash
set -x #echo on

gcloud compute ssh --zone=$'us-central1-b' jupyter@$'keras' -- -L 6006:localhost:6006