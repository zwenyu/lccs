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

cd ..

if [ $DATASET = "pacs" ]; then

	ND=2
	BATCH=128

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

else
	if [ $DATASET = "office31" ]; then
		ND=1
		BATCH=32
	elif [ $DATASET = 'office_home' ]; then
		ND=1
		BATCH=64
	fi

	python train.py \
	--root ${DATA} \
	--seed ${SEED} \
	--trainer ${TRAINER} \
	--source-domains ${S1} \
	--target-domains ${T} \
	--dataset-config-file configs/datasets/da/${DATASET}.yaml \
	--config-file configs/trainers/dg/mixstyle/${DATASET}.yaml \
	--output-dir ${OUTPUT_DIR}/${DATASET}/${TRAINER}_${NET}_ndomain${ND}_batch${BATCH}/${S1}/seed${SEED} \
	MODEL.BACKBONE.NAME ${NET} \
	DATALOADER.TRAIN_X.SAMPLER RandomDomainSampler \
	DATALOADER.TRAIN_X.N_DOMAIN ${ND} \
	DATALOADER.TRAIN_X.BATCH_SIZE ${BATCH} \
	OPTIM.MAX_EPOCH 50 \
	TRAIN.CHECKPOINT_FREQ 50

fi