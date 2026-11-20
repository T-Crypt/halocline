#!/usr/bin/env bash
# Builds three_way against release aa156951's own vision libraries (no change to Halocline). Output: /var/tmp/vision-ab/three_way
set -eu
R=$HOME/halocline/releases/aa156951
nvcc -std=c++20 -O2 -arch=sm_89 -I$R/include -I$R/src/vision "$(dirname "$0")/three_way.cu" -o /var/tmp/vision-ab/three_way \
  $R/build-89/tools/vision/libvision_cuda.a $R/build-89/tools/vision/libvision_core.a $R/build-89/libmodel_file.a $R/build-89/libplatform.a \
  -lcublas -lpthread
