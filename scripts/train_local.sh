#!/bin/bash
exp_name=${1}
# DATAROOT=../uorf_data_gen/image_generation/rendered_images/new_clevr_masked
PORT=6273
python -m visdom.server -p $PORT &>/dev/null &


## freeze test training

# DATAROOT=/media/mihir/dataset/uorf_dataset/clevr_567_test

# python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'clevr_567' \
#     --display_port $PORT --display_ncols 4 --display_freq 1 --display_grad \
#     --load_size 128 --n_samp 16 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 40 \
#     --num_slots 8  --display_env $exp_name --freeze_decoder --continue_train "main_model" --exp_id $exp_name\
#     --model 'uorf_nogan' --overfit --no_shuffle --lr_policy "constant" --lr 5e-5 --change_idx_after 5\



## freeze scene training

# DATAROOT=/media/mihir/dataset/uorf_dataset/clevr_567_train
# python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'clevr_567' \
#     --display_port $PORT --display_ncols 4 --print_freq 2 --display_freq 2 --display_grad \
#     --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 40 \
#     --num_slots 8 --no_mask --display_env $exp_name --freeze_decoder --continue_train "primitive_loaded" --exp_id $exp_name\
#     --model 'uorf_nogan' \


## primitive training

DATAROOT=../uorf_data_gen/image_generation/rendered_images/new_clevr
python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 3  \
    --checkpoints_dir 'checkpoints' --name 'clevr_567' \
    --display_port $PORT --display_ncols 4 --print_freq 2 --display_freq 2 --display_grad \
    --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 40 \
    --num_slots 2 --no_mask --display_env $exp_name --continue_train "main_model" --exp_id $exp_name --save_by_iter\
    --model 'uorf_nogan' \

# done
echo "Done"


# ipdb> rendered.shape
# torch.Size([4, 3, 64, 64])
# ipdb> z_slots.shape
# torch.Size([2, 40])
# ipdb> nss2cam0.shape
# torch.Size([1, 3, 3])
# ipdb> sampling_coor_fg.shape
# torch.Size([1, 1048576, 3])
