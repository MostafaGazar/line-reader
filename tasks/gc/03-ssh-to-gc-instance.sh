#!/bin/bash
set -x #echo on

gcloud compute ssh --zone=$'us-central1-b' jupyter@$'keras' -- -L 8080:localhost:8080