#!/bin/bash
exp_name=${1}

PORT=49159

python -m visdom.server -p $PORT  &>/dev/null &

# DATAROOT=/home/mprabhud/dataset/uorf/clevr_567_test

# python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'clevr_567' \
#     --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 1 --display_grad \
#     --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 40 \
#     --num_slots 8  --display_env $exp_name --freeze_decoder --continue_train "full_clevr_frozen" --exp_id $exp_name\
#     --model 'uorf_nogan' --no_shuffle --lr_policy "constant" --lr 1e-5 --change_idx_after 1000\

DATAROOT=/home/mprabhud/dataset/uorf/clevr_567_train

python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'clevr_567' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
    --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 40 \
    --num_slots 8 --no_mask --display_env $exp_name --freeze_decoder --epoch_count 212 --continue_train "full_clevr_frozen_2" --exp_id $exp_name --save_by_iter\
    --model 'uorf_nogan' \


# DATAROOT=/home/mprabhud/dataset/uorf/clevr_567_train

# python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'clevr_567' \
#     --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
#     --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 40 \
#     --num_slots 8 --no_mask --display_env $exp_name  --epoch_count 212 --continue_train "full_clevr2" --exp_id $exp_name --save_by_iter\
#     --model 'uorf_nogan' \


# done 
echo "Done"
