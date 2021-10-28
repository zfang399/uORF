#!/bin/bash
DATAROOT=${1:-'clevr_567_train'}
exp_name=${2}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 4 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'clevr_567' \
    --fixed_locality --display_port 49156 --display_ncols 4 --print_freq 2 --display_freq 2 --display_grad \
    --load_size 64 --n_samp 12 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 8 --num_slots 1 --no_mask --display_env $exp_name\
    --learn_masked --no_bkg --model 'uorf_nogan' \
# done
echo "Done"


# python train_without_gan.py --dataroot $DATAROOT --n_scenes 1000 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'clevr_567' \
#     --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
#     --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 40 --num_slots 8 \
#     --model 'uorf_nogan' \