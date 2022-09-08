#!/bin/bash

START=$1
END=$2
NET=$3
TRAINER=$4
OUTPUT_DIR=$5
NET_EVAL=$6
TRAINER_EVAL=$7
OUTPUT_DIR_PARENT=$8
OUTPUT_DIR_SUFFIX=$9

DATASET=pacs
D1=art_painting
D2=cartoon
D3=photo
D4=sketch

for SEED in $(seq ${START} ${END})
do
   	bash run_single_lccs.sh ${D2} ${D3} ${D4} ${D1} ${NET} ${SEED} ${DATASET} ${TRAINER} ${OUTPUT_DIR} ${NET_EVAL} ${TRAINER_EVAL} ${OUTPUT_DIR_PARENT} ${OUTPUT_DIR_SUFFIX}
   	# bash run_single_lccs.sh ${D1} ${D3} ${D4} ${D2} ${NET} ${SEED} ${DATASET} ${TRAINER} ${OUTPUT_DIR} ${NET_EVAL} ${TRAINER_EVAL} ${OUTPUT_DIR_PARENT} ${OUTPUT_DIR_SUFFIX}
   	# bash run_single_lccs.sh ${D1} ${D2} ${D4} ${D3} ${NET} ${SEED} ${DATASET} ${TRAINER} ${OUTPUT_DIR} ${NET_EVAL} ${TRAINER_EVAL} ${OUTPUT_DIR_PARENT} ${OUTPUT_DIR_SUFFIX}
   	# bash run_single_lccs.sh ${D1} ${D2} ${D3} ${D4} ${NET} ${SEED} ${DATASET} ${TRAINER} ${OUTPUT_DIR} ${NET_EVAL} ${TRAINER_EVAL} ${OUTPUT_DIR_PARENT} ${OUTPUT_DIR_SUFFIX}
done
