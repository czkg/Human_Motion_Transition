import torch
from .base_model import BaseModel
from . import networks
import numpy as np


class PathGANModel(BaseModel):
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
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
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
        self.z_size = opt.z_size
        self.masks = []

        #pretrained decoder

        #define mask
        for i in range(opt.path_length):
            mask = torch.zeros(self.input_size).float().to(self.device)
            mask[i * opt.input_latent : (i + 1) * opt.input_latent] = 1.0
            self.masks.append(mask)

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(self.input_size, self.output_size, self.z_size, num_downs=opt.num_downs,
                                      norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type, init_gain=opt.init_gain,
                                      gpu_ids=self.gpu_ids, where_add=opt.where_add)

        # define a discriminator. Conditional GANs need to take both input and putput. Therefore, #channels for D is input_nc + output_nc
        if self.isTrain:
            self.netD = networks.define_D(self.input_size + self.output_size, opt.output_latent, opt.d_layers, norm=opt.norm, nl=opt.nl,
                                          init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.BCELoss(reduction = 'sum')
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """ Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap paths in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)

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
        # self.loss_G_L1_0 = self.criterionL1(self.fake_B[:,:self.latent_size],self.real_B[:,:self.latent_size])*self.opt.lambda_L1
        # self.loss_G_L1_1 = self.criterionL1(self.fake_B[:,self.latent_size:2*self.latent_size], self.real_B[:,self.latent_size:2*self.latent_size])*self.opt.lambda_L1_inter
        # self.loss_G_L1_2 = self.criterionL1(self.fake_B[:,2*self.latent_size:3*self.latent_size], self.real_B[:,2*self.latent_size:3*self.latent_size])*self.opt.lambda_L1_inter
        # self.loss_G_L1_3 = self.criterionL1(self.fake_B[:,3*self.latent_size:4*self.latent_size], self.real_B[:,3*self.latent_size:4*self.latent_size])*self.opt.lambda_L1_inter
        # self.loss_G_L1_4 = self.criterionL1(self.fake_B[:,4*self.latent_size:5*self.latent_size], self.real_B[:,4*self.latent_size:5*self.latent_size])*self.opt.lambda_L1_inter
        # self.loss_G_L1_5 = self.criterionL1(self.fake_B[:,5*self.latent_size:6*self.latent_size], self.real_B[:,5*self.latent_size:6*self.latent_size])*self.opt.lambda_L1_inter
        # self.loss_G_L1_6 = self.criterionL1(self.fake_B[:,6*self.latent_size:7*self.latent_size], self.real_B[:,6*self.latent_size:7*self.latent_size])*self.opt.lambda_L1_inter
        # self.loss_G_L1_7 = self.criterionL1(self.fake_B[:,7*self.latent_size:8*self.latent_size], self.real_B[:,7*self.latent_size:8*self.latent_size])*self.opt.lambda_L1_inter
        # self.loss_G_L1_8 = self.criterionL1(self.fake_B[:,8*self.latent_size:9*self.latent_size], self.real_B[:,8*self.latent_size:9*self.latent_size])*self.opt.lambda_L1_inter
        # self.loss_G_L1_9 = self.criterionL1(self.fake_B[:,9*self.latent_size:], self.real_B[:,9*self.latent_size:])*self.opt.lambda_L1
        # self.loss_G_L1 = torch.tensor(0.0).to(self.device)
        # for i in range(len(self.masks)):
        #     self.loss_G_L1 += self.criterionL1(self.fake_B * self.masks[i], self.real_B * self.masks[i])
        # self.loss_G_L1 *= self.opt.lambda_L1
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1#_0 + self.loss_G_L1_1 + self.loss_G_L1_2 + self.loss_G_L1_3 + self.loss_G_L1_4 + self.loss_G_L1_5 + self.loss_G_L1_6 + self.loss_G_L1_7 + self.loss_G_L1_8 + self.loss_G_L1_9
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
