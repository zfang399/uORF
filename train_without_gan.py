import time
import ipdb
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import re
import numpy as np
import os
from util.visualizer import Visualizer
# from . import util
from util.util import AverageMeter, set_seed
from util import util
st = ipdb.set_trace


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    all_recon_losses = []
    before_adaptation_ari = []
    post_adaptation_ari = []
    all_aris = []

    # Set random seed for this experiment
    set_seed(opt.seed)


    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        print('Dataset size:', len(dataset))
        meters_trn = {stat: AverageMeter() for stat in model.loss_names}
        # st()
        opt.stage = 'coarse' if epoch < opt.coarse_epoch else 'fine'
        model.netDecoder.locality = True if epoch < opt.no_locality_epoch else False

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            # st()
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            if opt.sparse_nerf:
                layers, avg_grad = model.optimize_parameters(opt.display_grad, epoch, (total_iters-1)%opt.display_freq==0 )   # calculate loss functions, get gradients, update network weights
            else:
                layers, avg_grad = model.optimize_parameters_full(opt.display_grad, epoch)

            if opt.custom_lr and opt.stage == 'coarse':
                model.update_learning_rate()    # update learning rates at the beginning of every step

            # st()
            if (total_iters-1) % opt.display_freq == 0:   # display images on visdom and save images to a HTML file

                custom_iters = total_iters -1
                # st()
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()

                losses = model.get_current_losses()


                if opt.no_mask:
                    for loss_name in model.loss_names:
                        meters_trn[loss_name].update(float(losses[loss_name]))
                        losses[loss_name] = meters_trn[loss_name].avg
                else:
                    if opt.change_idx_after !=-1:
                        all_recon_losses.append(losses['recon'])
                        all_aris.append(losses['ari'])

                        if losses['ari'] >= max(all_aris):
                            best_ari_mask = model.render_mask0
                            best_rgb = model.x_rec0
                            
                        
                        if losses['recon'] <= min(all_recon_losses):
                            probable_best_ari = losses['ari']

                        if (((custom_iters+1) % opt.change_idx_after) == 0):

                            image_numpy = util.tensor2im(best_ari_mask)
                            img_path = os.path.join(visualizer.img_dir, 'epoch%.3d_%s.png' % (custom_iters, "best_mask"))
                            util.save_image(image_numpy, img_path)

                            image_numpy = util.tensor2im(best_rgb)
                            img_path = os.path.join(visualizer.img_dir, 'epoch%.3d_%s.png' % (custom_iters, "best_rgb"))
                            util.save_image(image_numpy, img_path)
                            
                            # st()
                            # print("ended")
                            # about to finish
                            post_adaptation_ari.append(probable_best_ari)
                            all_recon_losses = []
                            all_aris = []
                            probable_best_ari = 0.0
                            best_ari_mask =None
                            best_rgb = None

                        elif (custom_iters % opt.change_idx_after) == 0:
                            # st()
                            # print("started")
                            # about to start
                            before_adaptation_ari.append(losses['ari'])
                        # else:
                        #     print("nothing")
                    # st()
                    print("\n")
                    print("Pre Adaptation Avg:", np.mean(before_adaptation_ari), "Post Adaptation Avg:", np.mean(post_adaptation_ari))
                    print("\n")

                if opt.display_grad:
                    visualizer.display_grad(layers, avg_grad)


                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                # st()
                t_comp = (time.time() - iter_start_time) / opt.batch_size

                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                print('learning rate:', model.optimizers[0].param_groups[0]['lr'])



            # st()

            # if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            # st()
            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
                if opt.save_by_iter:
                    digit_regex = re.compile('\d+')
                    import glob 
                    new_names = model.model_names + ["lr_scheduler_0","optimizer_0"]
                    for model_name in new_names:
                        files = glob.glob(f'{model.save_dir}/iter*{model_name}.pth')
                        old_checkpoints = sorted([int(digit_regex.findall(file.split("/")[-1])[0]) for file in files])
                        if len(old_checkpoints) > 3:
                            if model_name in ["lr_scheduler_0","optimizer_0"]:
                                filename = f'{model.save_dir}/iter_{old_checkpoints[0]}_{model_name}.pth'
                            else:
                                filename = f'{model.save_dir}/iter_{old_checkpoints[0]}_net_{model_name}.pth'
                            os.remove(filename)                    



            iter_data_time = time.time()

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        if not opt.custom_lr:
            model.update_learning_rate()  # update learning rates at the end of every epoch.


