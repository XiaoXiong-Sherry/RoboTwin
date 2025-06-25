#!/bin/bash

policy_name=pi0_beat # [TODO] 
task_name=${1}
task_config=${2}
ckpt_setting=${3}
seed=${4}
gpu_id=${5}
# PI0 Model Parameters
model_path=${6:-"my_policies/pretrained_model-pi0_beat"}  # Default path to your model
pi0_step=${7:-1}  # Default number of action steps

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mmodel path: ${model_path}\033[0m"
echo -e "\033[33mpi0 step: ${pi0_step}\033[0m"

cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --model_path ${model_path} \
    --gpu_id ${gpu_id} \
    --pi0_step ${pi0_step}
