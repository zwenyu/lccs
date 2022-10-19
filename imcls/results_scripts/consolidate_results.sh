#!/bin/bash

cd /DATA/lccs/imcls

# PACS

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


# Office31

# declare -a source_model_list=("Vanilla_resnet50_ndomain1_batch32")
# declare -a source_output_dir_list=("output_source_models")
# declare -a lccs_output_dir_list=("output_results/LCCSLineark5n155/output_source_models_LCCSLineark5n155_resnet50_lccs_")

# dataset="office31"
# # declare -a test_envs=("amazon_webcam" "amazon_dslr" "webcam_amazon" "webcam_dslr" "dslr_amazon" "dslr_webcam")
# declare -a test_envs=("amazon_webcam" "amazon_dslr")

# # lccs adapted model
# for output_dir in "${lccs_output_dir_list[@]}"; do
# 	for source_model in "${source_model_list[@]}"; do
# 		for env in "${test_envs[@]}"; do	 
# 	    	python -m utils.consolidate_results --output_dir $output_dir --dataset $dataset --source_model $source_model --test_env $env --num_seed 3
# 	    done
# 	done
# done


# OfficeHome

# declare -a source_model_list=("Vanilla_resnet50_ndomain1_batch64")
# declare -a source_output_dir_list=("output_source_models")
# declare -a lccs_output_dir_list=(
# 	"output_results/LCCSLineark5n325/output_source_models_LCCSLineark5n325_resnet50_lccs_")

# dataset="office_home"
# # declare -a test_envs=("art_clipart" "art_product" "art_real_world" "clipart_art" "clipart_product" "clipart_real_world" "product_art" "product_clipart" "product_real_world" \
# # 	"real_world_art" "real_world_clipart" "real_world_product")
# declare -a test_envs=("art_clipart" "art_product" "art_real_world")

# # lccs adapted model
# for output_dir in "${lccs_output_dir_list[@]}"; do
# 	for source_model in "${source_model_list[@]}"; do
# 		for env in "${test_envs[@]}"; do	 
# 	    	python -m utils.consolidate_results --output_dir $output_dir --dataset $dataset --source_model $source_model --test_env $env --num_seed 3
# 	    done
# 	done
# done