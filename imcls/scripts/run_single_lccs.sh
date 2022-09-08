#!/bin/bash

DATA=/DATA/lccs/imcls/data

S1=$1
S2=$2
S3=$3
T=$4
NET=$5
SEED=$6
DATASET=$7
TRAINER=$8
OUTPUT_DIR=$9
NET_EVAL=${10}
TRAINER_EVAL=${11}
OUTPUT_DIR_PARENT=${12}
OUTPUT_DIR_SUFFIX=${13}

ND=2
BATCH=128

cd ..

python evaluate_lccs.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER_EVAL} \
--source-domains ${S1} ${S2} ${S3} \
--target-domains ${T} \
--dataset-config-file configs/datasets/dg/${DATASET}.yaml \
--config-file configs/trainers/dg/mixstyle/${DATASET}.yaml \
--output-dir ${OUTPUT_DIR_PARENT}/${OUTPUT_DIR}_${TRAINER_EVAL}_${NET_EVAL}_${OUTPUT_DIR_SUFFIX}/${DATASET}/${TRAINER}_${NET}_ndomain${ND}_batch${BATCH}/${T}/seed${SEED} \
--model-dir ${OUTPUT_DIR}/${DATASET}/${TRAINER}_${NET}_ndomain${ND}_batch${BATCH}/${T}/seed${SEED} \
--load-epoch 150 \
MODEL.BACKBONE.NAME ${NET_EVAL} \
OPTIM.MAX_EPOCH 150 \
TRAIN.CHECKPOINT_FREQ 150 \
DATALOADER.TRAIN_X.SAMPLER RandomSampler \
DATALOADER.TRAIN_X.BATCH_SIZE 32 \
