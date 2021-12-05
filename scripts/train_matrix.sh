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



# DATAROOT=/home/mprabhud/dataset/uorf/clevr_567_test

# python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'clevr_567' \
#     --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 1 --display_grad \
#     --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 40 \
#     --num_slots 8  --display_env $exp_name --freeze_decoder --continue_train "full_clevr_frozen" --exp_id $exp_name\
#     --model 'uorf_nogan' --no_shuffle --lr_policy "constant" --lr 1e-5 --change_idx_after 1000\

# DATAROOT=/home/mprabhud/dataset/uorf/clevr_567_train

# python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'clevr_567' \
#     --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
#     --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 40 \
#     --num_slots 8 --no_mask --display_env $exp_name --freeze_decoder --epoch_count 212 --continue_train "full_clevr_frozen" --exp_id $exp_name\
#     --model 'uorf_nogan' \


# DATAROOT=/home/mprabhud/dataset/uorf/clevr_567_train

# python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'clevr_567' \
#     --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
#     --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 40 \
#     --num_slots 8 --no_mask --display_env $exp_name  --continue_train "full_clevr_frozen" --exp_id $exp_name\
#     --model 'uorf_nogan' \

## primitive training

# DATAROOT=../uorf_data_gen/room_diverse/datasets/custom_renders/

# python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'room_diverse' \
#     --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
#     --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 64 \
#     --num_slots 2 --no_mask --display_env $exp_name --continue_train "main_model" --exp_id $exp_name --bottom\
#     --model 'uorf_nogan' \



# DATAROOT=../uorf_data_gen/room_diverse/datasets/custom_renders_bkg3/

# python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'room_diverse' \
#     --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
#     --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 100 --z_dim 64 \
#     --num_slots 2 --no_mask --display_env $exp_name --continue_train "primitive_room_bkg3" --exp_id $exp_name --bottom\
#     --model 'uorf_nogan' \


# DATAROOT=../uorf_data_gen/room_diverse/datasets/custom_renders_bkg3_primitives_fix//

# python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'room_diverse' \
#     --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
#     --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 64 \
#     --num_slots 2 --no_mask --display_env $exp_name --continue_train "primitive_room_bkg3" --exp_id $exp_name --bottom --epoch_count 40 --save_by_iter\
#     --model 'uorf_nogan' \



# DATAROOT=../uorf_data_gen/room_diverse/datasets/custom_renders_bkg3_primitives_fix//

# python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'room_diverse' \
#     --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
#     --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 64 \
#     --num_slots 2 --no_mask --display_env $exp_name --continue_train "primitive_room_bkg3" --exp_id $exp_name --bottom --epoch_count 40 --save_by_iter\
#     --model 'uorf_nogan' \


# baseline

# DATAROOT=../uorf_data_gen/room_diverse/datasets/custom_renders_bkg3_678//

# python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'room_diverse' \
#     --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
#     --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 64 \
#     --num_slots 9 --no_mask --display_env $exp_name  --exp_id $exp_name --bottom --save_by_iter --continue_train "baseline_room_678_1" --load_iter 170000 --epoch_count 85\
#     --model 'uorf_nogan' \

# ours no freezing

# DATAROOT=../uorf_data_gen/room_diverse/datasets/custom_renders_bkg3_678//

# python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'room_diverse' \
#     --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
#     --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 64 \
#     --num_slots 9 --no_mask --display_env $exp_name  --continue_train "complex_room_578" --load_iter 170000 --exp_id $exp_name --bottom --save_by_iter --epoch_count 85\
#     --model 'uorf_nogan' \


# ours no freezing

# DATAROOT=../uorf_data_gen/room_diverse/datasets/custom_renders_bkg3_678//

# DATAROOT=/home/mprabhud/dataset/uorf/clevr_567_test

# python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'room_diverse' \
#     --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
#     --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 64 \
#     --num_slots 9 --no_mask --display_env $exp_name  --continue_train "complex_room_678_freeze" --exp_id $exp_name --bottom --save_by_iter --freeze_decoder  --load_iter 245000 --epoch_count 122\
#     --model 'uorf_nogan' \


DATAROOT=../uorf_data_gen/room_diverse/datasets/custom_renders_bkg3_789_tests//
# DATAROOT=../uorf_data_gen/room_diverse/datasets/custom_renders_bkg3_789_tests//


# python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'room_diverse' \
#     --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 1 --display_grad \
#     --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 64 \
#     --num_slots 10 --display_env $exp_name  --continue_train "complex_room_678_freeze_1" --exp_id $exp_name --bottom --save_by_iter \
#     --freeze_decoder  --load_iter 485000 --model 'uorf_nogan' --no_shuffle --lr_policy "constant" --lr 1e-5 --change_idx_after 300\


#baseline 1 

# python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'room_diverse' \
#     --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 1 --display_grad \
#     --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 64 \
#     --num_slots 10 --display_env $exp_name  --continue_train "baseline_room_678_3" --exp_id $exp_name --bottom --save_by_iter \
#     --freeze_decoder  --load_iter 410000 --model 'uorf_nogan' --no_shuffle --lr_policy "constant" --lr 1e-5 --change_idx_after 300\


# #baseline 2

# python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'room_diverse' \
#     --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 1 --display_grad \
#     --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 64 \
#     --num_slots 10 --display_env $exp_name  --continue_train "baseline_sym" --exp_id $exp_name --bottom --save_by_iter \
#     --freeze_decoder  --load_iter 155000 --model 'uorf_nogan' --no_shuffle --lr_policy "constant" --lr 1e-5 --change_idx_after 300\



# #baseline 3


# python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'room_diverse' \
#     --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 1 --display_grad \
#     --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 64 \
#     --num_slots 10 --display_env $exp_name  --continue_train "baseline_sym_freeze" --exp_id $exp_name --bottom --save_by_iter \
#     --load_iter 145000 --model 'uorf_nogan' --no_shuffle --lr_policy "constant" --lr 1e-5 --change_idx_after 300\





# DATAROOT=../uorf_data_gen/room_diverse/datasets/custom_renders_bkg3_789_tests//

# python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'room_diverse' \
#     --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 1 --display_grad \
#     --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 64 \
#     --num_slots 10 --display_env $exp_name  --continue_train "complex_room_578_1" --exp_id $exp_name --bottom --save_by_iter \
#     --load_iter 410000 --model 'uorf_nogan' --no_shuffle --lr_policy "constant" --lr 1e-5 --change_idx_after 300\


# DATAROOT=/home/mprabhud/dataset/uorf/clevr_567_test

# python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'clevr_567' \
#     --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 1 --display_grad \
#     --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 40 \
#     --num_slots 8  --display_env $exp_name --freeze_decoder --continue_train "full_clevr_frozen" --exp_id $exp_name\
#     --model 'uorf_nogan' --no_shuffle --lr_policy "constant" --lr 1e-5 --change_idx_after 1000\


# DATAROOT=../uorf_data_gen/room_diverse/datasets/custom_renders/

# python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'room_chair' \
#     --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
#     --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 64 \
#     --num_slots 2 --no_mask --display_env $exp_name --continue_train "main_model" --exp_id $exp_name\
#     --model 'uorf_nogan' \


## whole training with frozen

# python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'clevr_567' \
#     --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
#     --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 40 \
#     --num_slots 2 --no_mask --display_env $exp_name --continue_train "primitive_loaded" --exp_id $exp_name\
#     --model 'uorf_nogan' \
# DATAROOT=../uorf_data_gen/room_diverse/datasets/custom_renders_bkg3_678_lots/

# python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'room_diverse' \
#     --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
#     --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 64 \
#     --num_slots 9 --no_mask --display_env $exp_name  --exp_id $exp_name --bottom --save_by_iter\
#     --model 'uorf_nogan' \

# DATAROOT=../uorf_data_gen/room_diverse/datasets/custom_renders_bkg3_678_lots/

# python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'room_diverse' \
#     --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
#     --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 64 \
#     --num_slots 9 --no_mask --display_env $exp_name  --exp_id $exp_name --bottom --save_by_iter --freeze_decoder\
#     --model 'uorf_nogan' \
# done 
echo "Done"