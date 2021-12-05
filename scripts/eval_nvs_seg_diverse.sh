# #!/bin/bash
# DATAROOT=${1:-'room_diverse_test'}
# CHECKPOINT=${2:-'./'}
# PORT=8077
# python -m visdom.server -p $PORT &>/dev/null &
# python test.py --dataroot $DATAROOT --n_scenes 500 --n_img_each_scene 4 \
#     --checkpoints_dir $CHECKPOINT --name room_diverse_models --exp_id latest --results_dir 'results' \
#     --display_port $PORT --display_ncols 4 \
#     --load_size 128 --input_size 128 --render_size 8 --frustum_size 128 --bottom \
#     --n_samp 256 --z_dim 64 --num_slots 5 \
#     --model 'uorf_eval'
# echo "Done"


exp_name=${1}

PORT=49158

python -m visdom.server -p $PORT  &>/dev/null &
# custom_renders_bkg3_678
DATAROOT=../uorf_data_gen/room_diverse/datasets/custom_renders_bkg3_678_12//

python test.py --dataroot $DATAROOT --n_scenes 500 --n_img_each_scene 12 \
    --checkpoints_dir 'checkpoints' --name 'room_diverse' --exp_id $exp_name --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --input_size 128 --render_size 8 --frustum_size 128 --bottom --continue_train "complex_room_678_freeze_1"\
    --n_samp 256 --z_dim 64 --num_slots 9 --mask_size 128\
    --model 'uorf_eval' --load_iter 485000 --save_by_iter --display_env $exp_name



# python train_without_gan.py --dataroot $DATAROOT --n_scenes 17999 --n_img_each_scene 4  \
#     --checkpoints_dir 'checkpoints' --name 'room_diverse' \
#     --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 1 --display_grad \
#     --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 64 \
#     --num_slots 10 --display_env $exp_name  --continue_train "complex_room_678_freeze_1" --exp_id $exp_name --bottom --save_by_iter \
#     --freeze_decoder  --load_iter 485000 --model 'uorf_nogan' --no_shuffle --lr_policy "constant" --lr 1e-5 --change_idx_after 300\
