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

ND=2
BATCH=128

cd ..

python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--source-domains ${S1} ${S2} ${S3} \
--target-domains ${T} \
--dataset-config-file configs/datasets/dg/${DATASET}.yaml \
--config-file configs/trainers/dg/mixstyle/${DATASET}.yaml \
--output-dir ${OUTPUT_DIR}/${DATASET}/${TRAINER}_${NET}_ndomain${ND}_batch${BATCH}/${T}/seed${SEED} \
MODEL.BACKBONE.NAME ${NET} \
DATALOADER.TRAIN_X.SAMPLER RandomDomainSampler \
DATALOADER.TRAIN_X.N_DOMAIN ${ND} \
DATALOADER.TRAIN_X.BATCH_SIZE ${BATCH} \
OPTIM.MAX_EPOCH 150 \
TRAIN.CHECKPOINT_FREQ 150