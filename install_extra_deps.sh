#!/bin/bash

# 安装额外的依赖包

# 安装 cuml-cu11
pip install --extra-index-url https://pypi.nvidia.com cuml-cu11

# 安装 dinov2
pip install --no-deps git+https://github.com/facebookresearch/dinov2.git@2302b6bf46953431b969155307b9bed152754069
