#!/bin/bash

# install dassl
# cd /DATA/Dassl.pytorch
# pip install -r requirements.txt --user
# python setup.py develop

cd /DATA/lccs/imcls/scripts

# PACS

# source classifier
bash run_batch_lccs.sh 1 5 resnet18 Vanilla output_source_models resnet18_lccs LCCSk1n7 output_results/LCCSk1n7 '' pacs
# bash run_batch_lccs.sh 1 5 resnet18 Vanilla output_source_models resnet18_lccs LCCSk5n35 output_results/LCCSk5n35 '' pacs
# bash run_batch_lccs.sh 1 5 resnet18 Vanilla output_source_models resnet18_lccs LCCSk10n70 output_results/LCCSk10n70 '' pacs

# mean centroid classifier
# bash run_batch_lccs.sh 1 5 resnet18 Vanilla output_source_models resnet18_lccs LCCSCentroidk1n7 output_results/LCCSCentroidk1n7 '' pacs
bash run_batch_lccs.sh 1 5 resnet18 Vanilla output_source_models resnet18_lccs LCCSCentroidk5n35 output_results/LCCSCentroidk5n35 '' pacs
bash run_batch_lccs.sh 1 5 resnet18 Vanilla output_source_models resnet18_lccs LCCSCentroidk10n70 output_results/LCCSCentroidk10n70 '' pacs

# Office31
# bash run_batch_lccs.sh 1 3 resnet50 Vanilla output_source_models resnet50_lccs LCCSLineark5n155 output_results/LCCSLineark5n155 '' office31

# OfficeHome
# bash run_batch_lccs.sh 1 3 resnet50 Vanilla output_source_models resnet50_lccs LCCSLineark5n325 output_results/LCCSLineark5n325 '' office_home