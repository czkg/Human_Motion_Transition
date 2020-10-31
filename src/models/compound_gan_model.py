import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import os


class CompoundGANModel(BaseModel):
    """ This class implements the PathGAN model, for learning a mapping from input paths to output paths given paired data.
    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        # parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        # parser.set_defaults(where_add='input', nz=0)
        # if is_train:
        #   parser.set_defaults(gan_mode='vanilla', lambda_l1=100.0)
        return parser

    def __init__(self, opt):
        """ Initialize the pathGan class.
        Parameters:
            opt (Option class)-- stores all the experiment flags, needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_BCE', 'G2_BCE', 'D_real', 'D_fake']
        # specify the paths you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training /test scripts will call <BaseMoel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        self.input_size = opt.input_latent * opt.path_length
        self.output_size = opt.output_latent * opt.path_length
        self.latent_size = opt.input_latent
        self.skeleton_size = opt.skeleton_size
        self.z_size = opt.z_size

        #pretrained decoder
        x_dim = opt.dim_heatmap ** 2 * opt.num_joints + opt.dim_heatmap * opt.num_joints
        self.netVAE = networks.VAE(x_dim, opt.z_dim, opt.pca_dim, True)
        self.netVAE = networks.init_net(self.netVAE, init_type = opt.init_type, init_gain = opt.init_gain, gpu_ids = opt.gpu_ids)

        load_filename = '%s_net_%s.pth' % (opt.epoch2, 'VAE')
        load_path = os.path.join(opt.checkpoints_dir, 'vae', load_filename)
        if isinstance(self.netVAE, torch.nn.DataParallel):
            self.netVAE = self.netVAE.module
        state_dict = torch.load(load_path, map_location=str(self.device))
        self.netVAE.load_state_dict(state_dict)
        for param in self.netVAE.parameters():
            param.requires_grad = False
        

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(self.input_size, self.output_size, self.z_size, num_downs=opt.num_downs,
                                      norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type, init_gain=opt.init_gain,
                                      gpu_ids=self.gpu_ids, where_add=opt.where_add)

        # define a discriminator. Conditional GANs need to take both input and output. Therefore, #channels for D is input_nc + output_nc
        if self.isTrain:
            self.netD = networks.define_D(self.input_size + self.output_size, opt.output_latent, opt.d_layers, norm=opt.norm, nl=opt.nl,
                                          init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            # for both parts
            self.criterionConsistency = networks.ConsistencyLoss().to(self.device)
            self.criterionBone = networks.BoneLoss().to(self.device)
            #self.criterionKeypose = networks.KeyposeLoss().to(self.device)
            self.criterionBCE = torch.nn.BCELoss(reduction = 'sum')
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input, dataset):
        """ Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        if dataset == 'path':
        	self.real_A = input['A'].to(self.device)
        	self.real_B = input['B'].to(self.device)
        if dataset == 'pose':
        	self.input_d = input['d'].to(self.device)
        	self.input_g = input['g'].to(self.device)

    def get_z_random(self, batch_size, z_size, random_type='gauss'):
        if random_type =='uni':
            z = torch.rand(batch_size, z_size) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(batch_size, z_size)
        return z.to(self.device)

    def forward(self):
        """ Run forward pass, called by both functions <optimize_parameters> and <test>.
        """
        if self.z_size > 0:
            self.z_random = self.get_z_random(self.real_A.size(0), self.z_size)
            self.fake_B = self.netG(self.real_A, self.z_random)
        else:
            self.fake_B = self.netG(self.real_A)  # G(A)

    def inference(self):
        with torch.no_grad():
            if self.z_size > 0:
                self.z_random = self.get_z_random(self.real_A.size(0), self.z_size)
                self.fake_B = self.netG(self.real_A, self.z_random)
            else:
                self.fake_B = self.netG(self.real_A)
            return self.fake_B

    def backward_D(self):
        """ Calculate GAN loss for the discriminator
        """
        prob = np.random.random(1)[0]
        # Fake; stop backprop to the generator by detaching fake_B
        # we use conditional GANs; we need to feed both input and output to the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        #shape_f_ab = fake_AB.shape
        #fake_AB += torch.rand(shape_f_ab).to(self.device) * 0.02
        pred_fake = self.netD(fake_AB.detach())
        if prob < 0.9:
            self.loss_D_fake = self.criterionGAN(pred_fake, False) * self.opt.lambda_GAN
        else:
            self.loss_D_fake = self.criterionGAN(pred_fake, True) * self.opt.lambda_GAN
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        #shape_r_ab = real_AB.shape
        #real_AB += torch.rand(shape_r_ab).to(self.device) * 0.02
        pred_real = self.netD(real_AB)
        if prob < 0.9:
            self.loss_D_real = self.criterionGAN(pred_real, True) * self.opt.lambda_GAN
        else:
            self.loss_D_real = self.criterionGAN(pred_real, False) * self.opt.lambda_GAN
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """ Calculate GAN and L1 loss for the generator
        """
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_GAN
        # Second, G(A) = B
        self.loss_G_BCE = self.criterionBCE(self.fake_B, self.real_B) * self.opt.lambda_BCE
        # Third, use decoder
        fake_B_decoded = self.netVAE(self.fake_B.view(self.fake_B.shape[0]*self.opt.path_length, self.latent_size))
        real_B_decoded = self.netVAE(self.real_B.view(self.real_B.shape[0]*self.opt.path_length, self.latent_size))
        self.loss_G2_BCE = self.criterionBCE(fake_B_decoded, real_B_decoded) * self.opt.lambda_BCE_decoder
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_BCE + self.loss_G2_BCE
        self.loss_G.backward()

    def update_G(self):
        # update G
        # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def update_D(self):
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights

    def optimize_parameters(self):
        self.forward()                   # compute fake paths: G(A)
        self.update_D()
        self.update_G()
