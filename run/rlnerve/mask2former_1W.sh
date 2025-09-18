#!/usr/bin/bash

cd /mmsegmentation-main/tools

# V100 32G
bash dist_test.sh ../myconfigs/rlnerve/dataset_paper/d3p_r34_e100.py ../workdir/deeplabv3plus/d3p_r34_e100/best_mIoU_epoch_1.pth 2