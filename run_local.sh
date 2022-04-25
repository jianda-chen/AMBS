#!/bin/bash

DOMAIN=finger
TASK=spin
ACTION_REPEAT=2
# DOMAIN=cartpole
# TASK=swingup
# ACTION_REPEAT=8
# DOMAIN=walker
# TASK=walk
# ACTION_REPEAT=2
# DOMAIN=cheetah
# TASK=run
# ACTION_REPEAT=4
# DOMAIN=reacher
# TASK=easy
# DOMAIN=ball_in_cup
# TASK=catch

SAVEDIR=../log

echo ${SAVEDIR}/${DOMAIN}_${TASK}_$(date +"%Y-%m-%d-%H-%M-%S")

MUJOCO_GL="egl" LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl/:$LD_LIBRARY_PATH python train.py \
    --domain_name ${DOMAIN} \
    --task_name ${TASK} \
    --agent 'ambs' \
    --init_steps 1000 \
    --num_train_steps 1000000 \
    --encoder_type pixel \
    --decoder_type identity \
    --transition_model_type 'probabilistic' \
    --img_source video \
    --resource_files '../kinetics-downloader/dataset/train/driving_car/*.mp4' \
    --action_repeat ${ACTION_REPEAT} \
    --critic_tau 0.01 \
    --encoder_tau 0.05 \
    --hidden_dim 1024 \
    --encoder_feature_dim 100 \
    --total_frames 1000 \
    --num_layers 4 \
    --num_filters 32 \
    --batch_size 128 \
    --encoder_lr 5e-4 \
    --decoder_lr 5e-4 \
    --actor_lr 5e-4 \
    --critic_lr 5e-4 \
    --alpha_lr 1e-4 \
    --alpha_beta 0.5 \
    --init_temperature 0.1 \
    --num_eval_episodes 5 \
    --work_dir ${SAVEDIR}/${DOMAIN}_${TASK}_$(date +"%Y-%m-%d-%H-%M-%S") \
    --adaptive_ratio \
    --sep_rew_dyn \
    --sep_rew_ratio 0.5 \
    --deep_metric \
    --save_tb \
    --seed 1 $@
