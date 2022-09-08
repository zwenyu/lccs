#!/bin/bash

cd /DATA/lccs/imcls

declare -a source_model_list=("Vanilla_resnet18_ndomain2_batch128")
declare -a source_output_dir_list=("output_source_models")
declare -a lccs_output_dir_list=("output_results/LCCSk1n7/output_source_models_LCCSk1n7_resnet18_lccs_"
	"output_results/LCCSCentroidk5n35/output_source_models_LCCSCentroidk5n35_resnet18_lccs_"
	"output_results/LCCSCentroidk10n70/output_source_models_LCCSCentroidk10n70_resnet18_lccs_")

dataset="pacs"
# declare -a test_envs=("art_painting" "cartoon" "photo" "sketch")
declare -a test_envs=("art_painting")

# source model
for output_dir in "${source_output_dir_list[@]}"; do
	for source_model in "${source_model_list[@]}"; do
		for env in "${test_envs[@]}"; do	 
	    	python -m utils.consolidate_results --output_dir $output_dir --dataset $dataset --source_model $source_model --test_env $env --num_seed 5
	    done
	done
done

# lccs adapted model
for output_dir in "${lccs_output_dir_list[@]}"; do
	for source_model in "${source_model_list[@]}"; do
		for env in "${test_envs[@]}"; do	 
	    	python -m utils.consolidate_results --output_dir $output_dir --dataset $dataset --source_model $source_model --test_env $env --num_seed 5
	    done
	done
done