#!/bin/bash
set -x #echo on

gcloud compute ssh --zone=$'us-west2-b' jupyter@$'line-reader' -- -L 8080:localhost:8080