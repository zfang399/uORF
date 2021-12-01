from itertools import chain

import torch
from torch import nn, optim
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
import os
import ipdb
st = ipdb.set_trace
from util.util import AverageMeter
from sklearn.metrics import adjusted_rand_score
import time
import numpy as np
from .projection import Projection
from torchvision.transforms import Normalize
from .model import Encoder, Decoder, Decoder_woBkg, SlotAttention, SlotAttention_woBkg, get_perceptual_net, raw2outputs
import cv2

EPS = 1e-6

class uorfNoGanModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--num_slots', metavar='K', type=int, default=8, help='Number of supported slots')
        parser.add_argument('--z_dim', type=int, default=64, help='Dimension of individual z latent per slot')
        parser.add_argument('--attn_iter', type=int, default=3, help='Number of refine iteration in slot attention')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
        parser.add_argument('--nss_scale', type=float, default=7, help='Scale of the scene, related to camera matrix')
        parser.add_argument('--render_size', type=int, default=64, help='Shape of patch to render each forward process. Must be Frustum_size/(2^N) where N=0,1,..., Smaller values cost longer time but require less GPU memory.')
        parser.add_argument('--supervision_size', type=int, default=64)
        parser.add_argument('--obj_scale', type=float, default=4.5, help='Scale for locality on foreground objects')
        parser.add_argument('--n_freq', type=int, default=5, help='how many increased freq?')
        parser.add_argument('--n_samp', type=int, default=64, help='num of samp per ray')
        parser.add_argument('--n_layer', type=int, default=3, help='num of layers bef/aft skip link in decoder')
        parser.add_argument('--weight_percept', type=float, default=0.006)
        parser.add_argument('--percept_in', type=int, default=100)
        parser.add_argument('--no_locality_epoch', type=int, default=300)
        parser.add_argument('--bottom', action='store_true', help='one more encoder layer on bottom')
        parser.add_argument('--input_size', type=int, default=64)
        parser.add_argument('--frustum_size', type=int, default=64)
        parser.add_argument('--frustum_size_fine', type=int, default=128)
        parser.add_argument('--attn_decay_steps', type=int, default=2e5)
        parser.add_argument('--coarse_epoch', type=int, default=600)
        parser.add_argument('--near_plane', type=float, default=6)
        parser.add_argument('--far_plane', type=float, default=20)
        parser.add_argument('--learn_masked', action='store_true', help='operate on masked inputs')
        parser.add_argument('--no_bkg', action='store_true', help='operate on masked inputs')        
        parser.add_argument('--freeze_decoder', action='store_true', help='operate on masked inputs')        
        parser.add_argument('--fixed_locality', action='store_true', help='enforce locality in world space instead of transformed view space')
        parser.add_argument('--train_autoencode', action='store_true', help='enforce locality in world space instead of transformed view space')

        parser.set_defaults(batch_size=1, lr=3e-4, niter_decay=0,
                            dataset_mode='multiscenes', niter=1200, custom_lr=True, lr_policy='warmup')

        parser.set_defaults(exp_id='run-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S')))

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel

        if opt.no_mask:
            self.loss_names = ['recon', 'perc']
        else:
            self.loss_names = ['recon', 'perc', 'ari', 'fgari', 'nvari']

        # self.loss_names = ['ari', 'fgari', 'nvari']

        n = opt.n_img_each_scene
        if opt.no_mask:
            self.visual_names = ['x{}'.format(i) for i in range(n)] + \
                                ['x_rec{}'.format(i) for i in range(n)] + \
                                ['slot{}_view{}'.format(k, i) for k in range(opt.num_slots) for i in range(n)] + \
                                ['unmasked_slot{}_view{}'.format(k, i) for k in range(opt.num_slots) for i in range(n)] + \
                                ['slot{}_attn'.format(k) for k in range(opt.num_slots)]
        else:
            self.visual_names = ['x{}'.format(i) for i in range(n)] + \
                                ['x_rec{}'.format(i) for i in range(n)] + \
                                ['slot{}_view{}'.format(k, i) for k in range(opt.num_slots) for i in range(n)] + \
                                ['unmasked_slot{}_view{}'.format(k, i) for k in range(opt.num_slots) for i in range(n)] + \
                                ['slot{}_attn'.format(k) for k in range(opt.num_slots)] + \
                                ['gt_mask{}'.format(i) for i in range(n)] + \
                                ['render_mask{}'.format(i) for i in range(n)]
        # st()

        self.model_names = ['Encoder', 'SlotAttention', 'Decoder']
        self.perceptual_net = get_perceptual_net().cuda()
        self.vgg_norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        render_size = (opt.render_size, opt.render_size)
        frustum_size = [self.opt.frustum_size, self.opt.frustum_size, self.opt.n_samp]
        self.projection = Projection(device=self.device, nss_scale=opt.nss_scale,
                                     frustum_size=frustum_size, near=opt.near_plane, far=opt.far_plane, render_size=render_size)
        frustum_size_fine = [self.opt.frustum_size_fine, self.opt.frustum_size_fine, self.opt.n_samp]
        self.projection_fine = Projection(device=self.device, nss_scale=opt.nss_scale,
                                          frustum_size=frustum_size_fine, near=opt.near_plane, far=opt.far_plane, render_size=render_size)
        z_dim = opt.z_dim
        self.num_slots = opt.num_slots
        self.netEncoder = networks.init_net(Encoder(3, z_dim=z_dim, bottom=opt.bottom),
                                            gpu_ids=self.gpu_ids, init_type='normal')
        if self.opt.no_bkg:
            self.netSlotAttention = networks.init_net(
                SlotAttention_woBkg(num_slots=opt.num_slots, in_dim=z_dim, slot_dim=z_dim, iters=opt.attn_iter), gpu_ids=self.gpu_ids, init_type='normal')            
        else:
            self.netSlotAttention = networks.init_net(
                SlotAttention(num_slots=opt.num_slots, in_dim=z_dim, slot_dim=z_dim, iters=opt.attn_iter), gpu_ids=self.gpu_ids, init_type='normal')

        if self.opt.no_bkg:
            self.netDecoder = networks.init_net(Decoder_woBkg(n_freq=opt.n_freq, input_dim=6*opt.n_freq+3+z_dim, z_dim=opt.z_dim, n_layers=opt.n_layer,
                                                        locality_ratio=opt.obj_scale/opt.nss_scale, fixed_locality=opt.fixed_locality), gpu_ids=self.gpu_ids, init_type='xavier')            
        else:            
            self.netDecoder = networks.init_net(Decoder(n_freq=opt.n_freq, input_dim=6*opt.n_freq+3+z_dim, z_dim=opt.z_dim, n_layers=opt.n_layer,
                                                        locality_ratio=opt.obj_scale/opt.nss_scale, fixed_locality=opt.fixed_locality), gpu_ids=self.gpu_ids, init_type='xavier')
        # st()
        if opt.freeze_decoder:
            self.netDecoder.eval()
            self.set_requires_grad(self.netDecoder, False)
            
        if self.isTrain:  # only defined during training time
            self.optimizer = optim.Adam(chain(
                self.netEncoder.parameters(), self.netSlotAttention.parameters(), self.netDecoder.parameters()
            ), lr=opt.lr)
            self.optimizers = [self.optimizer]

        self.L2_loss = nn.MSELoss()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def sql2_on_axis(self,x, axis, keepdim=True):
        return torch.sum(x**2, axis, keepdim=keepdim)


    def l1_on_axis(self, x, axis, keepdim=True):
        return torch.sum(torch.abs(x), axis, keepdim=keepdim)

    def l2_on_axis(self, x, axis, keepdim=True):
        return torch.sqrt(EPS + self.sql2_on_axis(x, axis, keepdim=keepdim))

    def reduce_masked_mean(self, x, mask, dim=None, keepdim=False):
        # x and mask are the same shape
        # returns shape-1
        # axis can be a list of axes
        # st()
        # assert(x.size() == mask.size())
        prod = x*mask
        if dim is None:
            numer = torch.sum(prod)
            denom = EPS+torch.sum(mask)
        
        else:
            numer = torch.sum(prod, dim=dim, keepdim=keepdim)
            denom = EPS+torch.sum(mask, dim=dim, keepdim=keepdim)
            
        mean = numer/denom
        return mean


    def setup(self, opt):
        """Load and print networks; create schedulers
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_{}'.format(opt.load_iter) if opt.load_iter > 0 else opt.epoch
            # st()
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        # st()
        self.x = input['img_data'].to(self.device)
        self.cam2world = input['cam2world'].to(self.device)
        if not self.opt.fixed_locality:
            self.cam2world_azi = input['azi_rot'].to(self.device)

        if 'masks' in input:
            self.gt_masks = input['masks']
            self.mask_idx = input['mask_idx']
            self.fg_idx = input['fg_idx']
            self.obj_idxs = input['obj_idxs']  # NxKxHxW


    def forward(self, epoch=0):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        if not self.opt.no_mask:
            self.weight_percept = 0.0
        else:
            self.weight_percept = self.opt.weight_percept if epoch >= self.opt.percept_in else 0
        self.loss_recon = 0
        self.loss_perc = 0
        dev = self.x[0:1].device
        nss2cam0 = self.cam2world[0:1].inverse() if self.opt.fixed_locality else self.cam2world_azi[0:1].inverse()
        # st()
        # Encoding images
        feature_map = self.netEncoder(F.interpolate(self.x[0:1], size=self.opt.input_size, mode='bilinear', align_corners=False))  # BxCxHxW
        feat = feature_map.flatten(start_dim=2).permute([0, 2, 1])  # BxNxC

        # Slot Attention
        z_slots, attn = self.netSlotAttention(feat)  # 1xKxC, 1xKxN
        z_slots, attn = z_slots.squeeze(0), attn.squeeze(0)  # KxC, KxN
        K = attn.shape[0]

        cam2world = self.cam2world
        N = cam2world.shape[0]

        if self.opt.stage == 'coarse':
            frus_nss_coor, z_vals, ray_dir = self.projection.construct_sampling_coor(cam2world)
            # (NxDxHxW)x3, (NxHxW)xD, (NxHxW)x3
            x = F.interpolate(self.x, size=self.opt.supervision_size, mode='bilinear', align_corners=False)
            self.z_vals, self.ray_dir = z_vals, ray_dir
        else:
            W, H, D = self.opt.frustum_size_fine, self.opt.frustum_size_fine, self.opt.n_samp
            start_range = self.opt.frustum_size_fine - self.opt.render_size
            rs = self.opt.render_size
            frus_nss_coor, z_vals, ray_dir = self.projection_fine.construct_sampling_coor(cam2world)
            # (NxDxHxW)x3, (NxHxW)xD, (NxHxW)x3
            frus_nss_coor, z_vals, ray_dir = frus_nss_coor.view([N, D, H, W, 3]), z_vals.view([N, H, W, D]), ray_dir.view([N, H, W, 3])
            H_idx = torch.randint(low=0, high=start_range, size=(1,), device=dev)
            W_idx = torch.randint(low=0, high=start_range, size=(1,), device=dev)
            frus_nss_coor_, z_vals_, ray_dir_ = frus_nss_coor[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :], z_vals[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :], ray_dir[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :]
            frus_nss_coor, z_vals, ray_dir = frus_nss_coor_.flatten(0, 3), z_vals_.flatten(0, 2), ray_dir_.flatten(0, 2)
            x = self.x[:, :, H_idx:H_idx + rs, W_idx:W_idx + rs]
            self.z_vals, self.ray_dir = z_vals, ray_dir

        if self.opt.no_bkg:
            sampling_coor_fg = frus_nss_coor[None, ...].expand(K, -1, -1)  # (K-1)xPx3
            raws, masked_raws, unmasked_raws, masks = self.netDecoder(sampling_coor_fg, z_slots, nss2cam0)  # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x1
        else:
            sampling_coor_fg = frus_nss_coor[None, ...].expand(K - 1, -1, -1)  # (K-1)xPx3
            sampling_coor_bg = frus_nss_coor  # Px3
            raws, masked_raws, unmasked_raws, masks = self.netDecoder(sampling_coor_bg, sampling_coor_fg, z_slots, nss2cam0)  # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x1

        W, H, D = self.opt.supervision_size, self.opt.supervision_size, self.opt.n_samp
        raws = raws.view([N, D, H, W, 4]).permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
        masked_raws = masked_raws.view([K, N, D, H, W, 4])
        unmasked_raws = unmasked_raws.view([K, N, D, H, W, 4])
        rgb_map, _, _ = raw2outputs(raws, z_vals, ray_dir)
        # (NxHxW)x3, (NxHxW)
        rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
        x_recon = rendered * 2 - 1
        # st()
        if self.opt.learn_masked:
            gt_mask_obj = (1.0 - (x.sum(1) == -3.).unsqueeze(1).float())
            loss_im = self.l2_on_axis(x_recon - x, 1, keepdim=True)
            # try:
            if torch.sum(gt_mask_obj) < 1e-4:
                weight = 1.0
            else:
                weight = torch.sum(1.0 - gt_mask_obj)/torch.sum(gt_mask_obj)
            weighted_mask = (gt_mask_obj * weight)  + ((1.0 - gt_mask_obj))
            loss_vis = (weighted_mask)[0].permute(1,2,0).detach().cpu().repeat(1,1,3).numpy().astype(np.uint8)
            x_vis = ((x +1.0)*127.0)[0].permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
            # st()
            cv2.imwrite('image_weight.jpg',loss_vis)
            cv2.imwrite('image_x.jpg',x_vis)
            # self.loss_vis = loss_im *weighted_mask
            self.loss_recon = self.reduce_masked_mean(loss_im, weighted_mask)
        else:
            # st()
            # st()
            if not self.opt.no_mask:
                self.loss_recon = self.L2_loss(x_recon[:1], x[:1])
            else:
                self.loss_recon = self.L2_loss(x_recon, x)

        x_norm, rendered_norm = self.vgg_norm((x + 1) / 2), self.vgg_norm(rendered)
        rendered_feat, x_feat = self.perceptual_net(rendered_norm), self.perceptual_net(x_norm)
        self.loss_perc = self.weight_percept * self.L2_loss(rendered_feat, x_feat)

        with torch.no_grad():
            attn = attn.detach().cpu()  # KxN
            H_, W_ = feature_map.shape[2], feature_map.shape[3]
            # st()
            attn = attn.view(self.opt.num_slots, 1, H_, W_)
            if H_ != H:
                attn = F.interpolate(attn, size=[H, W], mode='bilinear')
            for i in range(self.opt.n_img_each_scene):
                setattr(self, 'x_rec{}'.format(i), x_recon[i])
                setattr(self, 'x{}'.format(i), x[i])
            setattr(self, 'masked_raws', masked_raws.detach())
            setattr(self, 'unmasked_raws', unmasked_raws.detach())
            setattr(self, 'attn', attn)

    def compute_visuals(self):
        with torch.no_grad():
            _, N, D, H, W, _ = self.masked_raws.shape
            masked_raws = self.masked_raws  # KxNxDxHxWx4
            unmasked_raws = self.unmasked_raws  # KxNxDxHxWx4
            mask_maps = []
            for k in range(self.num_slots):
                raws = masked_raws[k]  # NxDxHxWx4
                z_vals, ray_dir = self.z_vals, self.ray_dir
                raws = raws.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
                
                if self.opt.no_mask:
                    rgb_map, depth_map, _ = raw2outputs(raws, z_vals, ray_dir)
                else:
                    rgb_map, depth_map, _, mask_map = raw2outputs(raws, z_vals, ray_dir,render_mask = True)
                    mask_maps.append(mask_map.view(N, H, W))

                rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
                x_recon = rendered * 2 - 1
                for i in range(self.opt.n_img_each_scene):
                    setattr(self, 'slot{}_view{}'.format(k, i), x_recon[i])

                raws = unmasked_raws[k]  # (NxDxHxW)x4
                raws = raws.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
                rgb_map, depth_map, _ = raw2outputs(raws, z_vals, ray_dir)
                rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
                x_recon = rendered * 2 - 1
                for i in range(self.opt.n_img_each_scene):
                    setattr(self, 'unmasked_slot{}_view{}'.format(k, i), x_recon[i])

                setattr(self, 'slot{}_attn'.format(k), self.attn[k] * 2 - 1)

            if not self.opt.no_mask:
                mask_maps = torch.stack(mask_maps)  # KxNxDxHxWx4
                mask_idx = mask_maps.cpu().argmax(dim=0)  # NxHxW

                # st()

                predefined_colors = []
                obj_idxs = self.obj_idxs  # Kx1xHxW
                gt_mask0 = self.gt_masks[0]  # 3xHxW
                for k in range(self.num_slots):
                    mask_idx_this_slot = mask_idx[0:1] == k  # 1xHxW
                    iou_this_slot = []
                    for kk in range(self.num_slots):
                        try:
                            obj_idx = obj_idxs[kk, ...]  # 1xHxW
                        except IndexError:
                            break
                        # st()
                        iou = (obj_idx & mask_idx_this_slot).type(torch.float).sum() / (obj_idx | mask_idx_this_slot).type(torch.float).sum()
                        iou_this_slot.append(iou)
                    target_obj_number = torch.tensor(iou_this_slot).argmax()
                    target_obj_idx = obj_idxs[target_obj_number, ...].squeeze()  # HxW
                    obj_first_pixel_pos = target_obj_idx.nonzero()[0]  # 2
                    obj_color = gt_mask0[:, obj_first_pixel_pos[0], obj_first_pixel_pos[1]]
                    predefined_colors.append(obj_color)
                predefined_colors = torch.stack(predefined_colors).permute([1,0])
                mask_visuals = predefined_colors[:, mask_idx]  # 3xNxHxW

                nvari_meter = AverageMeter()
                for i in range(N):
                    setattr(self, 'render_mask{}'.format(i), mask_visuals[:, i, ...])
                    setattr(self, 'gt_mask{}'.format(i), self.gt_masks[i])

                    this_mask_idx = mask_idx[i].flatten(start_dim=0)
                    gt_mask_idx = self.mask_idx[i]  # HW
                    fg_idx = self.fg_idx[i]
                    fg_idx_map = fg_idx.view([self.opt.frustum_size, self.opt.frustum_size])[None, ...]

                    fg_map = mask_visuals[0:1, i, ...].clone()
                    fg_map[fg_idx_map] = -1.
                    fg_map[~fg_idx_map] = 1.                    
                    setattr(self, 'bg_map{}'.format(i), fg_map)

                    if i == 0:
                        ari_score = adjusted_rand_score(gt_mask_idx, this_mask_idx)
                        fg_ari = adjusted_rand_score(gt_mask_idx[fg_idx], this_mask_idx[fg_idx])
                        self.loss_ari = ari_score
                        self.loss_fgari = fg_ari
                    else:
                        ari_score = adjusted_rand_score(gt_mask_idx, this_mask_idx)
                        nvari_meter.update(ari_score)
                    self.loss_nvari = nvari_meter.val                    
                # st()
                print(f"Different views ari: {self.loss_nvari}")
                print(f"Same view ari: {self.loss_ari}")
                print(f"Same view fg ari: {self.loss_fgari}")





    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        loss = self.loss_recon + self.loss_perc
        loss.backward()
        self.loss_perc = self.loss_perc / self.weight_percept if self.weight_percept > 0 else self.loss_perc

    def optimize_parameters(self, ret_grad=False, epoch=0):
        """Update network weights; it will be called in every training iteration."""
        self.forward(epoch)
        for opm in self.optimizers:
            opm.zero_grad()
        self.backward()
        avg_grads = []
        layers = []
        if ret_grad:
            for n, p in chain(self.netEncoder.named_parameters(), self.netSlotAttention.named_parameters(), self.netDecoder.named_parameters()):
                if p.grad is not None and "bias" not in n:
                    with torch.no_grad():
                        layers.append(n)
                        avg_grads.append(p.grad.abs().mean().cpu().item())
        for opm in self.optimizers:
            opm.step()
        return layers, avg_grads

    def save_networks(self, surfix):
        """Save all the networks to the disk.

        Parameters:
            surfix (int or str) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        super().save_networks(surfix)
        for i, opm in enumerate(self.optimizers):
            save_filename = '{}_optimizer_{}.pth'.format(surfix, i)
            save_path = os.path.join(self.save_dir, save_filename)
            torch.save(opm.state_dict(), save_path)

        for i, sch in enumerate(self.schedulers):
            save_filename = '{}_lr_scheduler_{}.pth'.format(surfix, i)
            save_path = os.path.join(self.save_dir, save_filename)
            torch.save(sch.state_dict(), save_path)

    def load_networks(self, surfix):
        """Load all the networks from the disk.

        Parameters:
            surfix (int or str) -- current epoch; used in he file name '%s_net_%s.pth' % (epoch, name)
        """
        super().load_networks(surfix)
        # st()

        if self.isTrain:
            for i, opm in enumerate(self.optimizers):
                load_filename = '{}_optimizer_{}.pth'.format(surfix, i)
                load_path = os.path.join(self.save_dir, load_filename)
                if os.path.exists(load_path):
                    print('loading the optimizer from %s' % load_path)
                    state_dict = torch.load(load_path, map_location=str(self.device))
                    opm.load_state_dict(state_dict)

            for i, sch in enumerate(self.schedulers):
                load_filename = '{}_lr_scheduler_{}.pth'.format(surfix, i)
                load_path = os.path.join(self.save_dir, load_filename)
                if os.path.exists(load_path):
                    print('loading the lr scheduler from %s' % load_path)
                    state_dict = torch.load(load_path, map_location=str(self.device))
                    sch.load_state_dict(state_dict)


if __name__ == '__main__':
    pass