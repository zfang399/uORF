#!/bin/bash
exp_name=${1}

PORT=49159

python -m visdom.server -p $PORT  &>/dev/null &
# primitive training

DATAROOT=../uorf_data_gen/room_diverse/datasets/custom_renders_bkg3_primitives_fix//

python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'room_diverse' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
    --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 64 \
    --num_slots 2 --no_mask --display_env $exp_name --continue_train "primitive_room_bkg3" --exp_id $exp_name --bottom --epoch_count 40 --save_by_iter\
    --model 'uorf_nogan' \


# 678 comiplex

DATAROOT=../uorf_data_gen/room_diverse/datasets/custom_renders_bkg3_678//

python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'room_diverse' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
    --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 64 \
    --num_slots 9 --no_mask --display_env $exp_name  --exp_id $exp_name --bottom --save_by_iter\
    --model 'uorf_nogan' \

# slow inference

DATAROOT=../uorf_data_gen/room_diverse/datasets/custom_renders_bkg3_789_tests//

python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'room_diverse' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 1 --display_grad \
    --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 64 \
    --num_slots 10 --display_env $exp_name  --continue_train "complex_room_678_freeze_1" --exp_id $exp_name --bottom --save_by_iter \
    --freeze_decoder  --load_iter 485000 --model 'uorf_nogan' --no_shuffle --lr_policy "constant" --lr 1e-5 --change_idx_after 300\


echo "Done"