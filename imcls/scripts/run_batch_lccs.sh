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
DATASET=${10}

if [ $DATASET = "pacs" ]; then
	
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

elif [ $DATASET = "office31" ]; then

	D1=amazon
	D2=webcam
	D3=dslr

	for SEED in $(seq ${START} ${END})
	do
	   	bash run_single_lccs.sh ${D1} None None ${D2} ${NET} ${SEED} ${DATASET} ${TRAINER} ${OUTPUT_DIR} ${NET_EVAL} ${TRAINER_EVAL} ${OUTPUT_DIR_PARENT} ${OUTPUT_DIR_SUFFIX}
	   	bash run_single_lccs.sh ${D1} None None ${D3} ${NET} ${SEED} ${DATASET} ${TRAINER} ${OUTPUT_DIR} ${NET_EVAL} ${TRAINER_EVAL} ${OUTPUT_DIR_PARENT} ${OUTPUT_DIR_SUFFIX}
	   	# bash run_single_lccs.sh ${D2} None None ${D1} ${NET} ${SEED} ${DATASET} ${TRAINER} ${OUTPUT_DIR} ${NET_EVAL} ${TRAINER_EVAL} ${OUTPUT_DIR_PARENT} ${OUTPUT_DIR_SUFFIX}
	   	# bash run_single_lccs.sh ${D2} None None ${D3} ${NET} ${SEED} ${DATASET} ${TRAINER} ${OUTPUT_DIR} ${NET_EVAL} ${TRAINER_EVAL} ${OUTPUT_DIR_PARENT} ${OUTPUT_DIR_SUFFIX}
	   	# bash run_single_lccs.sh ${D3} None None ${D1} ${NET} ${SEED} ${DATASET} ${TRAINER} ${OUTPUT_DIR} ${NET_EVAL} ${TRAINER_EVAL} ${OUTPUT_DIR_PARENT} ${OUTPUT_DIR_SUFFIX}
	   	# bash run_single_lccs.sh ${D3} None None ${D2} ${NET} ${SEED} ${DATASET} ${TRAINER} ${OUTPUT_DIR} ${NET_EVAL} ${TRAINER_EVAL} ${OUTPUT_DIR_PARENT} ${OUTPUT_DIR_SUFFIX}
	done

elif [ $DATASET = "office_home" ]; then

	D1=art
	D2=clipart
	D3=product
	D4=real_world

	for SEED in $(seq ${START} ${END})
	do
	   	bash run_single_lccs.sh ${D1} None None ${D2} ${NET} ${SEED} ${DATASET} ${TRAINER} ${OUTPUT_DIR} ${NET_EVAL} ${TRAINER_EVAL} ${OUTPUT_DIR_PARENT} ${OUTPUT_DIR_SUFFIX}
	   	bash run_single_lccs.sh ${D1} None None ${D3} ${NET} ${SEED} ${DATASET} ${TRAINER} ${OUTPUT_DIR} ${NET_EVAL} ${TRAINER_EVAL} ${OUTPUT_DIR_PARENT} ${OUTPUT_DIR_SUFFIX}
	   	bash run_single_lccs.sh ${D1} None None ${D4} ${NET} ${SEED} ${DATASET} ${TRAINER} ${OUTPUT_DIR} ${NET_EVAL} ${TRAINER_EVAL} ${OUTPUT_DIR_PARENT} ${OUTPUT_DIR_SUFFIX}
	   	# bash run_single_lccs.sh ${D2} None None ${D1} ${NET} ${SEED} ${DATASET} ${TRAINER} ${OUTPUT_DIR} ${NET_EVAL} ${TRAINER_EVAL} ${OUTPUT_DIR_PARENT} ${OUTPUT_DIR_SUFFIX}
	   	# bash run_single_lccs.sh ${D2} None None ${D3} ${NET} ${SEED} ${DATASET} ${TRAINER} ${OUTPUT_DIR} ${NET_EVAL} ${TRAINER_EVAL} ${OUTPUT_DIR_PARENT} ${OUTPUT_DIR_SUFFIX}
	   	# bash run_single_lccs.sh ${D2} None None ${D4} ${NET} ${SEED} ${DATASET} ${TRAINER} ${OUTPUT_DIR} ${NET_EVAL} ${TRAINER_EVAL} ${OUTPUT_DIR_PARENT} ${OUTPUT_DIR_SUFFIX}
	   	# bash run_single_lccs.sh ${D3} None None ${D1} ${NET} ${SEED} ${DATASET} ${TRAINER} ${OUTPUT_DIR} ${NET_EVAL} ${TRAINER_EVAL} ${OUTPUT_DIR_PARENT} ${OUTPUT_DIR_SUFFIX}
	   	# bash run_single_lccs.sh ${D3} None None ${D2} ${NET} ${SEED} ${DATASET} ${TRAINER} ${OUTPUT_DIR} ${NET_EVAL} ${TRAINER_EVAL} ${OUTPUT_DIR_PARENT} ${OUTPUT_DIR_SUFFIX}
	   	# bash run_single_lccs.sh ${D3} None None ${D4} ${NET} ${SEED} ${DATASET} ${TRAINER} ${OUTPUT_DIR} ${NET_EVAL} ${TRAINER_EVAL} ${OUTPUT_DIR_PARENT} ${OUTPUT_DIR_SUFFIX}
	   	# bash run_single_lccs.sh ${D4} None None ${D1} ${NET} ${SEED} ${DATASET} ${TRAINER} ${OUTPUT_DIR} ${NET_EVAL} ${TRAINER_EVAL} ${OUTPUT_DIR_PARENT} ${OUTPUT_DIR_SUFFIX}
	   	# bash run_single_lccs.sh ${D4} None None ${D2} ${NET} ${SEED} ${DATASET} ${TRAINER} ${OUTPUT_DIR} ${NET_EVAL} ${TRAINER_EVAL} ${OUTPUT_DIR_PARENT} ${OUTPUT_DIR_SUFFIX}
	   	# bash run_single_lccs.sh ${D4} None None ${D3} ${NET} ${SEED} ${DATASET} ${TRAINER} ${OUTPUT_DIR} ${NET_EVAL} ${TRAINER_EVAL} ${OUTPUT_DIR_PARENT} ${OUTPUT_DIR_SUFFIX}
	done

fi