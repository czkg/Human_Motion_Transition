import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler

###############################################################################
# Helper functions
###############################################################################


def init_weights(net, init_type='normal', init_gain=1.):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'zeros':
                init.zeros_(m.weight.data)
            elif init_type == 'ones':
                init.ones_(m.weight.data)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=1., gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions.
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def get_norm_layer(norm_type='none'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm1d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm1d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


def define_G(input_size, output_size, z_size, num_downs=7, norm='none', nl='relu',
             use_dropout=False, init_type='xavier', init_gain=0.02, gpu_ids=[], where_add='none'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl_layer = get_non_linearity(layer_type=nl)

    if z_size == 0:
        where_add = 'none'

    if where_add == 'none':
        net = G_Unet(input_size, output_size, num_downs, norm_layer=norm_layer, nl_layer=nl_layer,
                     use_dropout=use_dropout)
    elif where_add == 'input':
        net = G_Unet_add_input(input_size, output_size, z_size, num_downs, norm_layer=norm_layer, nl_layer=nl_layer,
                               use_dropout=use_dropout)
    elif where_add == 'all':
        net = G_Unet_add_all(input_size, output_size, z_size, num_downs, norm_layer=norm_layer, nl_layer=nl_layer,
                             use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net)

    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_size, n_neurons, d_layers=3, norm='none', nl='lrelu', init_type='xavier', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl = 'lrelu'  # use leaky relu for D
    nl_layer = get_non_linearity(layer_type=nl)

    net = D_NLayers(input_size, n_neurons, n_layers=d_layers, nl_layer=nl_layer, norm_layer=norm_layer)
    
    return init_net(net, init_type, init_gain, gpu_ids)


def define_E(input_size, output_size, ndf, netE,
             norm='batch', nl='lrelu',
             init_type='xavier', init_gain=0.02, gpu_ids=[], vaeLike=False):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl = 'lrelu'  # use leaky relu for E
    nl_layer = get_non_linearity(layer_type=nl)
    if netE == 'resnet_128':
        net = E_ResNet(input_size, output_size, ndf, n_blocks=4, norm_layer=norm_layer,
                       nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'resnet_256':
        net = E_ResNet(input_size, output_size, ndf, n_blocks=5, norm_layer=norm_layer,
                       nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'conv_128':
        net = E_NLayers(input_size, output_size, ndf, n_layers=4, norm_layer=norm_layer,
                        nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'conv_256':
        net = E_NLayers(input_size, output_size, ndf, n_layers=5, norm_layer=norm_layer,
                        nl_layer=nl_layer, vaeLike=vaeLike)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % net)

    return init_net(net, init_type, init_gain, gpu_ids)


def heatmap2pose(heatmap):
    dim_heatmap = 64
    n_joints = 17
    dim_xy = dim_heatmap ** 2
    dim_z = dim_heatmap
    #dim = dim_xy + dim_heatmap

    l = torch.linspace(-1, 1, dim_heatmap)
    data = heatmap.view(heatmap.shape[0], n_joints, -1)
    data_xy, data_z = torch.split(data, [dim_xy, dim_z], dim=2)

    xy_max = torch.argmax(data_xy, dim=-1)
    x_max = xy_max // dim_heatmap
    y_max = xy_max % dim_heatmap
    z_max = torch.argmax(data_z, dim=-1)
    x = l[x_max]
    y = l[y_max]
    z = l[z_max]

    pose = torch.stack((x, y, z), dim=-1)
    return pose


# class D_NLayersMulti(nn.Module):
#     def __init__(self, input_size, n_layers=3,
#                  n_neurons=512, num_D=1):
#         super(D_NLayersMulti, self).__init__()
#         # st()
#         self.num_D = num_D
#         if num_D == 1:
#             layers = self.get_layers(input_size, n_layers, n_neurons)
#             self.model = nn.Sequential(*layers)
#         else:
#             layers = self.get_layers(input_size, n_layers, n_neurons)
#             self.add_module("model_0", nn.Sequential(*layers))
#             self.down = nn.AvgPool1d(3, stride=2, padding=[
#                                      1, 1], count_include_pad=False)
#             for i in range(1, num_D):
#                 layers = self.get_layers(input_size, n_layers, n_neurons)
#                 self.add_module("model_%d" % i, nn.Sequential(*layers))

#     def get_layers(self, input_size, n_layers=3, n_neurons=512):
#         sequence = []

#         for _ in range(n_layers):
#             sequence += [nn.Linear(input_size, n_neurons), nn.LeakyReLU(0.2, True)]

#         sequence += [nn.Linear(n_neurons, output_size)]

#         return sequence

#     def forward(self, input):
#         if self.num_D == 1:
#             return self.model(input)
#         result = []
#         down = input
#         for i in range(self.num_D):
#             model = getattr(self, "model_%d" % i)
#             result.append(model(down))
#             if i != self.num_D - 1:
#                 down = self.down(down)
#         return result


class D_NLayers(nn.Module):
    """Defines a GAN discriminator"""

    def __init__(self, input_size, n_neurons, n_layers=3, nl_layer=None, norm_layer=None):
        """Construct a GAN discriminator
        Parameters:
            input_size (int)  -- the size of inputs
            n_layers (int)  -- the number of linear layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(D_NLayers, self).__init__()
        # if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
        #     use_bias = norm_layer.func != nn.BatchNorm2d
        # else:
        #     use_bias = norm_layer != nn.BatchNorm2d
        relu = nl_layer()
        output_size = 1 # output dimension
        sequence = []
        sequence += ([nn.Linear(input_size, n_neurons)] + [relu])

        for _ in range(n_layers):
            sequence += ([nn.Linear(n_neurons, n_neurons)] + [relu])

        sequence += [nn.Linear(n_neurons, output_size)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


##############################################################################
# Classes
##############################################################################
class VAE(nn.Module):
    """ This class implements the VAE model, 
        for encoding the observation data into latent space 
        and recover the original data from it.
    """

    def __init__(self, x_dim, z_dim, pca_dim, is_decoder):
        """ Initialize the VAE class

        Parameters:
            opt (Option class) -- stores all the experiment flags, needs to be a subclass of BaseOptions
        """
        super(VAE, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.pca_dim =pca_dim
        self.is_decoder = is_decoder

        # build the network
        # encoder
        self.fc1 = nn.Linear(self.x_dim, self.pca_dim)
        self.fc2 = nn.Linear(self.pca_dim, self.pca_dim)
        self.fc3 = nn.Linear(self.pca_dim, self.pca_dim)
        self.fc41 = nn.Linear(self.pca_dim, self.z_dim)
        self.fc42 = nn.Linear(self.pca_dim, self.z_dim)
        # decoder
        self.fc5 = nn.Linear(self.z_dim, self.pca_dim)
        self.fc6 = nn.Linear(self.pca_dim, self.pca_dim)
        self.fc7 = nn.Linear(self.pca_dim, self.pca_dim)
        self.fc8 = nn.Linear(self.pca_dim, self.x_dim)


    def gen_pca(self, x):
        pca = sklearn.decomposition.PCA(n_components = self.pca_dim, whiten = False)
        pca.fit(x)
        self.pca_weights = pca.components_


    def encoder(self, x):
        #build encoder model
        h1 = F.leaky_relu(self.fc1(x))
        h2 = F.leaky_relu(self.fc2(h1))
        h3 = F.leaky_relu(self.fc3(h2))
        mu = self.fc41(h3)
        logvar = self.fc42(h3)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = torch.sigmoid(mu + eps * std)
        return z

    def decoder(self, z):
        h4 = F.leaky_relu(self.fc5(z))
        h5 = F.leaky_relu(self.fc6(h4))
        h6 = F.leaky_relu(self.fc7(h5))
        return torch.sigmoid(self.fc8(h6))


    def forward(self, x):
        if not self.is_decoder:
            mu, logvar = self.encoder(x)
            z = self.reparameterize(mu, logvar)
            out = self.decoder(z)
            return out, mu, logvar, z
        else:
            out = self.decoder(x)
            return out


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()

    def __call__(self, mu, logvar, inputs, outputs):
        kl_loss = torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss *= -0.5

        recons_loss = F.binary_cross_entropy(outputs, inputs, reduction = 'sum')
        #recons_loss *= 0.5

        loss = recons_loss + kl_loss
        #print(inputs, '----', outputs)
        return loss


class VAE2(nn.Module):
    """ This class implements the alternative VAE model, 
        for encoding the observation data into latent space 
        and recover the original data from it.
    """

    def __init__(self, x_dim, z_dim, pca_dim, is_decoder):
        """ Initialize the alternative VAE class

        Parameters:
            opt (Option class) -- stores all the experiment flags, needs to be a subclass of BaseOptions
        """
        super(VAE2, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.pca_dim =pca_dim
        self.is_decoder = is_decoder

        # build the network
        # encoder
        self.fc1 = nn.Linear(self.x_dim, self.pca_dim)
        self.fc2 = nn.Linear(self.pca_dim, self.pca_dim)
        self.fc3 = nn.Linear(self.pca_dim, 2*self.z_dim)
        # decoder
        self.fc4 = nn.Linear(self.z_dim, self.pca_dim)
        self.fc5 = nn.Linear(self.pca_dim, self.pca_dim)
        self.fc6 = nn.Linear(self.pca_dim, self.x_dim)


    def gen_pca(self, x):
        pca = sklearn.decomposition.PCA(n_components = self.pca_dim, whiten = False)
        pca.fit(x)
        self.pca_weights = pca.components_


    def encoder(self, x):
        #build encoder model
        h1 = F.leaky_relu(self.fc1(x))
        h2 = F.leaky_relu(self.fc2(h1))
        zd = self.fc3(h2)
        return zd

    def reparameterize(self, dist):
        mu, logvar = torch.split(dist, [self.z_dim, self.z_dim], dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = torch.sigmoid(mu + eps * std)
        return sample

    def decoder(self, z):
        h1 = F.leaky_relu(self.fc4(z))
        h2 = F.leaky_relu(self.fc5(h1))
        return torch.sigmoid(self.fc6(h2))


    def forward(self, x):
        if not self.is_decoder:
            zd = self.encoder(x)
            z = self.reparameterize(zd)
            out = self.decoder(z)
            return out, zd, z
        else:
            out = self.decoder(x)
            return out


class VAE2Loss(nn.Module):
    def __init__(self):
        super(VAE2Loss, self).__init__()

    def __call__(self, zd, inputs, outputs):
        dim = int(zd.shape[-1]/2)
        mu, logvar = torch.split(zd, [dim, dim], dim=-1)
        kl_loss = torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss *= -0.5

        recons_loss = F.binary_cross_entropy(outputs, inputs, reduction = 'sum')
        #recons_loss *= 0.5

        loss = recons_loss + kl_loss
        #print(inputs, '----', outputs)
        return loss


class VAE2D(nn.Module):
    """ This class implements the VAE2D model, 
        for encoding the observation data into latent space 
        and recover the original data from it.
    """

    def __init__(self, x_dim, z_dim, pca_dim, is_decoder):
        """ Initialize the VAE class

        Parameters:
            opt (Option class) -- stores all the experiment flags, needs to be a subclass of BaseOptions
        """
        super(VAE2D, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        in_channels = 1
        h_dims = [16, 32, 64, 128, 256]
        self.is_decoder = is_decoder

        # build the network
        # encoder
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=h_dims[0], kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=h_dims[0], out_channels=h_dims[1], kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=h_dims[1], out_channels=h_dims[2], kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=h_dims[2], out_channels=h_dims[3], kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels=h_dims[3], out_channels=h_dims[4], kernel_size=1)
        self.fc_mu = nn.Linear(self.x_dim * h_dims[-1], self.z_dim)
        self.fc_var = nn.Linear(self.x_dim * h_dims[-1], self.z_dim)
        # decoder
        self.fc_de = nn.Linear(self.z_dim, self.x_dim * h_dims[-1])
        self.deconv1 = nn.ConvTranspose2d(h_dims[4], h_dims[3], kernel_size=1)
        self.deconv2 = nn.ConvTranspose2d(h_dims[3], h_dims[2], kernel_size=1)
        self.deconv3 = nn.ConvTranspose2d(h_dims[2], h_dims[1], kernel_size=1)
        self.deconv4 = nn.ConvTranspose2d(h_dims[1], h_dims[0], kernel_size=1)
        self.final_de = nn.Conv2d(in_channels=h_dims[0], out_channels=in_channels, kernel_size=1)


    def gen_pca(self, x):
        pca = sklearn.decomposition.PCA(n_components = self.pca_dim, whiten = False)
        pca.fit(x)
        self.pca_weights = pca.components_


    def encoder(self, x):
        #build encoder model
        h1 = F.leaky_relu(self.conv1(x))
        h2 = F.leaky_relu(self.conv2(h1))
        h3 = F.leaky_relu(self.conv3(h2))
        h4 = F.leaky_relu(self.conv4(h3))
        h5 = F.leaky_relu(self.conv5(h4))
        h5 = torch.flatten(h5, start_dim=1)
        mu = self.fc_mu(h5)
        logvar = self.fc_var(h5)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = torch.sigmoid(mu + eps * std)
        return z

    def decoder(self, z):
        h6 = self.fc_de(z)
        h6 = h6.view(-1, 256, 24, 3)
        h7 = F.leaky_relu(self.deconv1(h6))
        h8 = F.leaky_relu(self.deconv2(h7))
        h9 = F.leaky_relu(self.deconv3(h8))
        h10 = F.leaky_relu(self.deconv4(h9))
        return torch.sigmoid(self.final_de(h10))


    def forward(self, x):
        if not self.is_decoder:
            mu, logvar = self.encoder(x)
            z = self.reparameterize(mu, logvar)
            out = self.decoder(z)
            return out, z, mu, logvar
        else:
            out = self.decoder(x)
            return out


class VAE2DLoss(nn.Module):
    def __init__(self):
        super(VAE2DLoss, self).__init__()

    def __call__(self, mu, logvar, inputs, outputs):
        kl_loss = torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss *= -0.5

        recons_loss = F.mse_loss(outputs, inputs, reduction = 'sum')*100
        #recons_loss *= 0.5

        loss = recons_loss + kl_loss
        #print(inputs, '----', outputs)
        return loss


class MVAE(nn.Module):
    def __init__(self):
        super(MVAE, self).__init__()

    def forward(self, x):
        return x
        

class RecLoss(nn.Module):
    def __init__(self, use_L2=True):
        super(RecLoss, self).__init__()
        self.use_L2 = use_L2

    def __call__(self, input, target, batch_mean=True):
        if self.use_L2:
            diff = (input - target) ** 2
        else:
            diff = torch.abs(input - target)
        if batch_mean:
            return torch.mean(diff)
        else:
            return torch.mean(torch.mean(torch.mean(diff, dim=1), dim=2), dim=3)



class ConsistencyLoss(nn.Module):
    def __init__(self):
        super(ConsistencyLoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def __call__(self, outputs, path_length):
        """Consistency loss call function

        Parameters:
            outputs:  sequence of generated n latent codes, of shape(b, n, m), b is batch size, n is the length of the path, m is the dimension of one latent code
        return:
            total_loss: shape of (b,)
        """
        out = outputs.view(outputs.shape[0], path_length, -1)
        out_prev = out[:,:-1,:]
        out_next = out[:,1:,:]

        loss = self.loss(out_next, out_prev)

        return loss

class ConsistencyLoss_poses(nn.Module):
    def __init__(self):
        super(ConsistencyLoss_poses, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def __call__(self, outputs, path_length):
        """Consistency loss call function

        Parameters:
            outputs:  sequence of generated n latent codes, of shape(b, n, m), b is batch size, n is the length of the path, m is the dimension of one latent code
        return:
            total_loss: shape of (b,)
        """
        out = outputs.view(outputs.shape[0]//path_length, path_length, -1, 3)
        out_prev = out[:,:-1,:,:]
        out_next = out[:,1:,:,:]

        loss = self.loss(out_next, out_prev)

        return loss

class KeyposeLoss(nn.Module):
    def __init__(self):
        super(KeyposeLoss, self).__init__()
        self.loss = torch.nn.MSELoss(reduction='sum')

    def __call__(self, inputs, targets, path_length):
        """Keypose loss call function

        Parameters:
            inputs: sequence of generated n latent codes, of shape(b, n, m), b is batch size, m is the dimension of one latent code
            targets:  sequence of generated n latent codes, of shape(b, n, m), b is batch size, m is the dimension of one latent code
        """
        inputs = inputs.view(inputs.shape[0], path_length, -1)
        targets = targets.view(targets.shape[0], path_length, -1)
        start_pose_i = inputs[:,0,:]
        start_pose_t = targets[:,0,:]
        end_pose_i = inputs[:,-1,:]
        end_pose_t = targets[:,-1,:]

        start_loss = self.loss(start_pose_i, start_pose_t)
        end_loss = self.loss(end_pose_i, end_pose_t)

        loss = start_loss + end_loss
        return loss


class BoneLoss(nn.Module):
    def __init__(self):
        super(BoneLoss, self).__init__()

    def __call__(self, outputs, inputs, body, path_length):
        """Bone loss call function, compare the bone size of predicted sequence from start pose in target

        Paramters:
            outputs:  sequence of n 3D poses, of shape (b, n, m). m is the number of joints.
            inputs: sequence of n 3D poses, of shape (b, n, m). m is the number of joints.
        """
        outputs = outputs.view(outputs.shape[0]//path_length, path_length, outputs.shape[1], -1)
        inputs = inputs.view(inputs.shape[0]//path_length, path_length, inputs.shape[1], -1)
        num_of_skeletons = outputs.shape[1]
        num_of_joints = outputs.shape[2]

        loss = 0.
        for skeleton_index in range(num_of_skeletons):
            for bone in body.bones:
                i = bone[0]
                j = bone[1]

                x1 = inputs[:,0,i,0]
                y1 = inputs[:,0,i,1]
                z1 = inputs[:,0,i,2]

                x2 = inputs[:,0,j,0]
                y2 = inputs[:,0,j,1]
                z2 = inputs[:,0,j,2]

                bone_length_ref = torch.sqrt(torch.pow((x2-x1),2) + torch.pow((y2-y1),2) + torch.pow((z2-z1),2))

                x1 = outputs[:,skeleton_index,i,0]
                y1 = outputs[:,skeleton_index,i,1]
                z1 = outputs[:,skeleton_index,i,2]

                x2 = outputs[:,skeleton_index,j,0]
                y2 = outputs[:,skeleton_index,j,1]
                z2 = outputs[:,skeleton_index,j,2]

                bone_length = torch.sqrt(torch.pow((x2-x1),2) + torch.pow((y2-y1),2) + torch.pow((z2-z1),2))

                loss += torch.sum(torch.abs(bone_length - bone_length_ref))

        return loss


# Defines the GAN loss which uses different GANs.
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real input
            target_fake_label (bool) - - label of a fake input

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgangp':
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, predictions, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor list) - - tpyically the prediction output from a discriminator; supports multi Ds.
            target_is_real (bool) - - if the ground truth label is for real paths or fake paths

        Returns:
            the calculated loss.
        """
        all_losses = []
        for prediction in predictions:
            if self.gan_mode in ['lsgan', 'vanilla']:
                target_tensor = self.get_target_tensor(prediction, target_is_real)
                loss = self.loss(prediction, target_tensor)
            elif self.gan_mode == 'wgangp':
                if target_is_real:
                    loss = -prediction.mean()
                else:
                    loss = prediction.mean()
            all_losses.append(loss)
        total_loss = sum(all_losses)
        return total_loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet.
class G_Unet(nn.Module):
    def __init__(self, input_size, output_size, num_downs,
                 norm_layer=None, nl_layer=None, use_dropout=False):
        super(G_Unet, self).__init__()
        # construct unet structure
        unet_block = UnetBlock(input_size // 16, input_size // 16, input_size // 16,
                               innermost=True, norm_layer=norm_layer, nl_layer=nl_layer)
        for i in range(num_downs - 5):
            unet_block = UnetBlock(input_size // 16, input_size // 16, input_size // 16, unet_block,
                                   norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout)
        unet_block = UnetBlock(input_size // 8, input_size // 8, input_size // 16, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer)
        unet_block = UnetBlock(input_size // 4, input_size // 4, input_size // 8, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer)
        unet_block = UnetBlock(input_size // 2, input_size // 2, input_size // 4, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer)
        unet_block = UnetBlock(input_size, output_size, input_size // 2, unet_block,
                               outermost=True, norm_layer=norm_layer, nl_layer=nl_layer)

        self.model = unet_block

    def forward(self, x):
        return self.model(x)


# Defines the Unet generator with z at input.
# |num_downs|: number of downsamplings in UNet.
class G_Unet_add_input(nn.Module):
    def __init__(self, input_size, output_size, z_size, num_downs,
                 norm_layer=None, nl_layer=None, use_dropout=False):
        super(G_Unet_add_input, self).__init__()
        self.z_size = z_size
        # construct unet structure
        unet_block = UnetBlock(input_size // 16, input_size // 16, input_size // 16,
                               innermost=True, norm_layer=norm_layer, nl_layer=nl_layer)
        for i in range(num_downs - 5):
            unet_block = UnetBlock(input_size // 16, input_size // 16, input_size // 16, unet_block,
                                   norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout)
        unet_block = UnetBlock(input_size // 8, input_size // 8, input_size // 16, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer)
        unet_block = UnetBlock(input_size // 4, input_size // 4, input_size // 8, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer)
        unet_block = UnetBlock(input_size // 2, input_size // 2, input_size // 4, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer)
        unet_block = UnetBlock(input_size + z_size, output_size, input_size // 2, unet_block,
                               outermost=True, norm_layer=norm_layer, nl_layer=nl_layer)

        self.model = unet_block

    def forward(self, x, z=None):
        if self.z_size > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(
                z.size(0), z.size(1), x.size(2), x.size(3))
            x_with_z = torch.cat([x, z_img], 1)
        else:
            x_with_z = x  # no z

        return self.model(x_with_z)



# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- encode -- |submodule| -- decode --|
# we use fully connected layers
class UnetBlock(nn.Module):
    def __init__(self, input_size, outer_size, inner_size,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        downlinear = []
        downlinear += [nn.Linear(input_size, inner_size)]
        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_size) if norm_layer is not None else None
        uprelu = nl_layer()
        upnorm = norm_layer(outer_size) if norm_layer is not None else None

        if outermost:
            uplinear = [nn.Linear(inner_size * 2, outer_size)]
            down = downlinear
            up = [uprelu] + uplinear + [nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            uplinear = [nn.Linear(inner_size, outer_size)]
            down = [downrelu] + downlinear
            up = [uprelu] + uplinear
            if upnorm is not None:
                up += [upnorm]
            model = down + up
        else:
            uplinear = [nn.Linear(inner_size * 2, outer_size)]
            down = [downrelu] + downlinear
            if downnorm is not None:
                down += [downnorm]
            up = [uprelu] + uplinear
            if upnorm is not None:
                up += [upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat((self.model(x), x), 1)


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=True)


# two usage cases, depend on kw and padw
def upsampleConv(inplanes, outplanes, kw, padw):
    sequence = []
    sequence += [nn.Upsample(scale_factor=2, mode='nearest')]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=kw,
                           stride=1, padding=padw, bias=True)]
    return nn.Sequential(*sequence)


def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes,
                           kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)


class BasicBlockUp(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlockUp, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [upsampleConv(inplanes, outplanes, kw=3, padw=1)]
        if norm_layer is not None:
            layers += [norm_layer(outplanes)]
        layers += [conv3x3(outplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = upsampleConv(inplanes, outplanes, kw=1, padw=0)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [conv3x3(inplanes, inplanes)]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class E_ResNet(nn.Module):
    def __init__(self, input_size=3, output_size=1, ndf=64, n_blocks=4,
                 norm_layer=None, nl_layer=None, vaeLike=False):
        super(E_ResNet, self).__init__()
        self.vaeLike = vaeLike
        max_ndf = 4
        conv_layers = [
            nn.Conv2d(input_size, ndf, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)
            output_ndf = ndf * min(max_ndf, n + 1)
            conv_layers += [BasicBlock(input_ndf,
                                       output_ndf, norm_layer, nl_layer)]
        conv_layers += [nl_layer(), nn.AvgPool2d(8)]
        if vaeLike:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_size)])
            self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_size)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_size)])
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        else:
            return output
        return output


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet.
class G_Unet_add_all(nn.Module):
    def __init__(self, input_size, output_size, z_size, num_downs,
                 norm_layer=None, nl_layer=None, use_dropout=False):
        super(G_Unet_add_all, self).__init__()
        # construct unet structure
        unet_block = UnetBlock_with_z(input_size // 16, input_size // 16, input_size // 16, z_size, None, innermost=True,
                                      norm_layer=norm_layer, nl_layer=nl_layer)
        for i in range(num_downs - 5):
            unet_block = UnetBlock_with_z(input_size // 16, input_size // 16, input_size // 16, z_size, unet_block,
                                          norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout)
        unet_block = UnetBlock_with_z(input_size // 8, input_size // 8, input_size // 16, z_size, unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer)
        unet_block = UnetBlock_with_z(input_size // 4, input_size // 4, input_size // 8, z_size, unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer)
        unet_block = UnetBlock_with_z(input_size // 2, input_size // 2, input_size // 4, z_size, unet_block, 
                                      norm_layer=norm_layer, nl_layer=nl_layer)
        unet_block = UnetBlock_with_z(input_size, output_size, input_size // 2, z_size, unet_block,
                                      outermost=True, norm_layer=norm_layer, nl_layer=nl_layer)
        self.model = unet_block

    def forward(self, x, z):
        return self.model(x, z)


class UnetBlock_with_z(nn.Module):
    def __init__(self, input_size, outer_size, inner_size, z_size=0,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False):
        super(UnetBlock_with_z, self).__init__()
        downlinear = []

        self.outermost = outermost
        self.innermost = innermost
        self.z_size = z_size
        input_size = input_size + z_size
        downlinear += [nn.Linear(input_size, inner_size)]
        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nl_layer()

        if outermost:
            uplinear = [nn.Linear(inner_size * 2, outer_size)]
            down = downlinear
            up = [uprelu] + uplinear + [nn.Sigmoid()]
        elif innermost:
            uplinear = [nn.Linear(inner_size, outer_size)]
            down = [downrelu] + downlinear
            up = [uprelu] + uplinear
            if norm_layer is not None:
                up += [norm_layer(outer_size)]
        else:
            uplinear = [nn.Linear(inner_size * 2, outer_size)]
            down = [downrelu] + downlinear
            if norm_layer is not None:
                down += [norm_layer(inner_size)]
            up = [uprelu] + uplinear

            if norm_layer is not None:
                up += [norm_layer(outer_size)]

            if use_dropout:
                up += [nn.Dropout(0.5)]
        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x, z):
        # print(x.size())
        if self.z_size > 0:
            x_and_z = torch.cat([x, z], 1)
        else:
            x_and_z = x

        if self.outermost:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return self.up(x2)
        elif self.innermost:
            x1 = self.up(self.down(x_and_z))
            return torch.cat([x1, x], 1)
        else:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return torch.cat([self.up(x2), x], 1)


class E_NLayers(nn.Module):
    def __init__(self, input_size, output_size=1, ndf=64, n_layers=3,
                 norm_layer=None, nl_layer=None, vaeLike=False):
        super(E_NLayers, self).__init__()
        self.vaeLike = vaeLike

        kw, padw = 4, 1
        sequence = [nn.Conv2d(input_size, ndf, kernel_size=kw,
                              stride=2, padding=padw), nl_layer()]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 4)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw)]
            if norm_layer is not None:
                sequence += [norm_layer(ndf * nf_mult)]
            sequence += [nl_layer()]
        sequence += [nn.AvgPool2d(8)]
        self.conv = nn.Sequential(*sequence)
        self.fc = nn.Sequential(*[nn.Linear(ndf * nf_mult, output_size)])
        if vaeLike:
            self.fcVar = nn.Sequential(*[nn.Linear(ndf * nf_mult, output_size)])

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        return output

###########################################################################

class VAEDMP(nn.Module):
    """Define deep variational bayes filters (DVBF) objectives.
    """

    def __init__(self, x_dim, u_dim, z_dim, hidden_dim, transform_dim, noise_dim, is_decoder, device, alpha=25.0, beta=25.0/4.0, tau=1.0, dt=0.01):
        """ Initialize the GANLoss class.
        Parameters:
            x_dim (int) - - dimension of observation x
            u_dim (int) - - dimension of control signal u
            z_dim (int) - - dimension of latent code z
            hidden_dim (int) - - dimension of hidden layers
        """
        super(VAEDMP, self).__init__()
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.z_dim = z_dim
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.transform_dim = transform_dim

        self.device = device

        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.dt = dt

        # encoder
        self.fc_en1 = nn.Linear(self.x_dim, self.hidden_dim)
        self.fc_en2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_en3 = nn.Linear(self.hidden_dim, self.z_dim)

        # initial enc for noise1
        self.fc_in1 = nn.Linear(self.z_dim, self.transform_dim)
        self.fc_in2 = nn.Linear(self.transform_dim, 2*self.noise_dim)
        # initial enc for z1
        self.fc_in3 = nn.Linear(self.noise_dim, self.transform_dim)
        self.fc_in4 = nn.Linear(self.transform_dim, self.z_dim)

        # decoder
        self.fc_de1 = nn.Linear(self.z_dim, self.hidden_dim)
        self.fc_de2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_de3 = nn.Linear(self.hidden_dim, self.x_dim)

        # noise_net
        self.fc_no1 = nn.Linear(2*self.z_dim+self.u_dim, self.transform_dim)
        self.fc_no2 = nn.Linear(self.transform_dim, 2*self.noise_dim)
        
        # force_net (use LSTM)
        #self.lstm = nn.LSTM(self.z_dim, self.u_dim)

    def A(self, zt, dzt):
        """Matrix A in transition model
        """
        I = torch.eye(2)

        A1 = (-1 * self.dt * self.alpha * self.beta * (1./self.tau)) * self.dt + 1.
        A2 = (-1 * self.dt * self.alpha * (1./self.tau) + 1.) * self.dt
        A3 = (-1 * self.alpha * self.beta * (1./self.tau)) * self.dt
        A4 = (-1 * self.alpha * (1./self.tau)) * self.dt + 1.

        a1 = A1 * zt + A2 * dzt
        a2 = A3 * zt + A4 * dzt

        aa = torch.stack((a1, a2), dim=0)
        return aa

    def b(self, z_goal, f, eps):
        """Matrix b in transition model
        """
        b = (self.alpha * self.beta * z_goal + f + eps) * self.dt * (1./self.tau)
        b1 = self.dt * b
        b2 = 1. * b

        bb = torch.stack((b1, b2), dim=0)
        return bb

    def encoder(self, x):
        """Encoder network to extract features from inputs X
        Parameters:
            x (tensor) -- inputs in observation space
        """
        h1 = F.leaky_relu(self.fc_en1(x))
        h2 = F.leaky_relu(self.fc_en2(h1))
        feat = F.leaky_relu(self.fc_en3(h2))
        return feat

    def init_enc(self, x):
        """Network to generate z1, w1 and z_goal, w_goal from features
        """
        h1 = F.leaky_relu(self.fc_in1(x))
        noise_dist = F.leaky_relu(self.fc_in2(h1))
        noise = self.sample(noise_dist)

        h2 = F.leaky_relu(self.fc_in3(noise))
        z = F.leaky_relu(self.fc_in4(h2))
        return noise_dist, noise, z


    def sample(self, dist):
        """Function to generate samples from distributions
        """
        mu, logvar = torch.split(dist, [self.noise_dim, self.noise_dim], dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = torch.sigmoid(mu + eps * std)
        return sample

    def decoder(self, z):
        """Decoder network to recover x from z
        """
        h1 = F.leaky_relu(self.fc_de1(z))
        h2 = F.leaky_relu(self.fc_de2(h1))
        return torch.sigmoid(self.fc_de3(h2))

    def noise_net(self, feature, z, f):
        """Network to generate wi (noise) from xi and z(i-1)
        Parameters:
            feature (tensor) -- inputs extract from encoder
            z (tensor) -- inputs in latent space
            f (tensor) -- force inputs
        """
        x = torch.cat((feature, z),dim=1)
        h = F.leaky_relu(self.fc_no1(torch.cat((x,f),dim=1)))
        noise_dist = F.leaky_relu(self.fc_no2(h))
        noise = self.sample(noise_dist)
        return noise_dist, noise

    # def phase(self, n_steps, t=None):
    #     """The phase variable replaces explicit timing.

    #     It starts with 1 at the begining of the movement and converges exponentially to 0.
    #     """
    #     phases = torch.exp(-self.alpha/3. * torch.linspace(0, 1, n_steps))
    #     if t is None:
    #         return phases
    #     else:
    #         return phases[t]

    def force(self, z, g):
        """Network to generate force for each frame
        Parameters:
            z (tensor) -- list of latent codes [len_sequence, batch, z_dim]
        Returns:
            f (tensor) -- force of current frame [batch, z_dim]
        """
        #zg = z[-1,...]
        dz = self.d(z)
        ddz = self.dd(z)

        f = self.tau*self.tau*ddz - self.alpha*(
            self.beta*(g - z) - self.tau*dz)
        f = torch.sigmoid(f)

        return f

    def d(self, x):
        d = torch.zeros_like(x)
        if x.shape[0] == 1:
            return d
        else:
            for i in range(x.shape[0]):
                if i == 0:
                    d[i,...] = x[i+1,...] - x[i,...]
                elif i == x.shape[0]-1:
                    d[i,...] = x[i,...] - x[i-1,...]
                else:
                    d[i,...] = x[i+1,...] - x[i-1,...] / 2.
            return d

    def dd(self, x):
        return self.d(self.d(x))


    def forward(self, x):
        """Calculate VAE-DMP's output. Take input as sequence.
        Parameters:
            x (tensor) - - observation inputs of shape [batch, len_sequence, x_dim]
        Returns:
            Latent codes zs and reconstructed xs.
        """
        # first, reshape x from [batch, len_sequence, x_dim] to [len_sequence, batch, x_dim]
        x = torch.transpose(x, 0, 1)
        # z1
        features = self.encoder(x)
        wd1, w1, _ = self.init_enc(features[0])
        z1 = features[0]
        x1 = self.decoder(z1)
        # z2
        # _, _, z2 = self.init_enc(features[1])
        z2 = features[1]
        # z_goal
        # _, _, zn = self.init_enc(features[-1])
        zn = features[-1]

        dz1 = (z2 - z1) / self.dt
        zt = z1.clone()
        dzt = dz1.clone()

        xs = x1.unsqueeze(0)
        zs = z1.unsqueeze(0)
        wds = wd1.unsqueeze(0)
        ws = w1.unsqueeze(0)
        for t, xt in enumerate(x[1:]):
            # calculate f
            fs = self.force(zs, zn)
            ft = fs[-1]
            wdt, wt = self.noise_net(features[t], zt, ft)
            wds = torch.cat((wds, wdt.unsqueeze(0)), 0)
            # calculate z[t+1]
            nex = self.A(zt, dzt) + self.b(zn, ft, wt)
            zt = nex[0]
            dzt = nex[1]
            zs = torch.cat((zs, zt.unsqueeze(0)), 0)
            # calculate reconstructed x
            xt = self.decoder(zt)
            xs = torch.cat((xs, xt.unsqueeze(0)), 0)

        xs = torch.transpose(xs, 0, 1)
        zs = torch.transpose(zs, 0, 1)
        wds = torch.transpose(wds, 0, 1)
        zzs = torch.transpose(features, 0, 1)
        fs = torch.transpose(fs, 0, 1)
        return xs, zs, zzs, wds, fs

###########################################################################


#Define DVBF module
# class VAEDMP(nn.Module):
#     """Define deep variational bayes filters (DVBF) objectives.
#     """

#     def __init__(self, x_dim, u_dim, z_dim, hidden_dim, transform_dim, noise_dim, is_decoder, device, alpha=25.0, beta=25.0/4.0, tau=1.0, dt=0.01):
#         """ Initialize the GANLoss class.

#         Parameters:
#             x_dim (int) - - dimension of observation x
#             u_dim (int) - - dimension of control signal u
#             z_dim (int) - - dimension of latent code z
#             hidden_dim (int) - - dimension of hidden layers
#         """
#         super(VAEDMP, self).__init__()
#         self.x_dim = x_dim
#         self.u_dim = u_dim
#         self.z_dim = z_dim
#         self.noise_dim = noise_dim
#         self.hidden_dim = hidden_dim
#         self.transform_dim = transform_dim

#         self.device = device
#         self.is_decoder = is_decoder

#         self.alpha = alpha
#         self.beta = beta
#         self.tau = tau
#         self.dt = dt

#         # encoder
#         self.fc_en1 = nn.Linear(self.x_dim, self.hidden_dim)
#         self.fc_en2 = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.fc_en3 = nn.Linear(self.hidden_dim, self.z_dim)

#         # initial enc for noise1
#         self.fc_in1 = nn.Linear(self.z_dim, self.transform_dim)
#         self.fc_in2 = nn.Linear(self.transform_dim, 2*self.noise_dim)
#         # initial enc for z1
#         self.fc_in3 = nn.Linear(self.noise_dim, self.transform_dim)
#         self.fc_in4 = nn.Linear(self.transform_dim, self.z_dim)

#         # decoder
#         self.fc_de1 = nn.Linear(self.z_dim, self.hidden_dim)
#         self.fc_de2 = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.fc_de3 = nn.Linear(self.hidden_dim, self.x_dim)

#         # noise_net
#         self.fc_no1 = nn.Linear(2*self.z_dim+self.u_dim, self.transform_dim)
#         self.fc_no2 = nn.Linear(self.transform_dim, 2*self.noise_dim)
        
#         # force_net (use LSTM)
#         #self.lstm = nn.LSTM(self.z_dim, self.u_dim)

#     def A(self, zt, dzt):
#         """Matrix A in transition model
#         """
#         I = torch.eye(2)

#         A1 = (-1 * self.dt * self.alpha * self.beta * (1./self.tau)) * self.dt + 1.
#         A2 = (-1 * self.dt * self.alpha * (1./self.tau) + 1.) * self.dt
#         A3 = (-1 * self.alpha * self.beta * (1./self.tau)) * self.dt
#         A4 = (-1 * self.alpha * (1./self.tau)) * self.dt + 1.

#         a1 = A1 * zt + A2 * dzt
#         a2 = A3 * zt + A4 * dzt

#         aa = torch.stack((a1, a2), dim=0)
#         return aa

#     def b(self, z_goal, f, eps):
#         """Matrix b in transition model
#         """
#         b = (self.alpha * self.beta * z_goal + f + eps) * self.dt * (1./self.tau)
#         b1 = self.dt * b
#         b2 = 1. * b

#         bb = torch.stack((b1, b2), dim=0)
#         return bb

#     def encoder(self, x):
#         """Encoder network to extract features from inputs X
#         Parameters:
#             x (tensor) -- inputs in observation space
#         """
#         h1 = F.leaky_relu(self.fc_en1(x))
#         h2 = F.leaky_relu(self.fc_en2(h1))
#         feat = F.leaky_relu(self.fc_en3(h2))
#         return feat

#     def init_enc(self, x):
#         """Network to generate z1, w1 and z_goal, w_goal from features
#         """
#         h1 = F.leaky_relu(self.fc_in1(x))
#         noise_dist = F.leaky_relu(self.fc_in2(h1))
#         noise = self.sample(noise_dist)

#         h2 = F.leaky_relu(self.fc_in3(noise))
#         z = F.leaky_relu(self.fc_in4(h2))
#         return noise_dist, noise, z


#     def sample(self, dist):
#         """Function to generate samples from distributions
#         """
#         mu, logvar = torch.split(dist, [self.noise_dim, self.noise_dim], dim=1)
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         sample = torch.sigmoid(mu + eps * std)
#         return sample

#     def decoder(self, z):
#         """Decoder network to recover x from z
#         """
#         h1 = F.leaky_relu(self.fc_de1(z))
#         h2 = F.leaky_relu(self.fc_de2(h1))
#         return torch.sigmoid(self.fc_de3(h2))

#     def noise_net(self, feature, z, f):
#         """Network to generate wi (noise) from xi and z(i-1)
#         Parameters:
#             feature (tensor) -- inputs extract from encoder
#             z (tensor) -- inputs in latent space
#         """
#         x = torch.cat((feature, z),dim=1)
#         h = F.leaky_relu(self.fc_no1(torch.cat((x,f),dim=1)))
#         noise_dist = F.leaky_relu(self.fc_no2(h))
#         noise = self.sample(noise_dist)
#         return noise_dist, noise

#     # def phase(self, n_steps, t=None):
#     #     """The phase variable replaces explicit timing.

#     #     It starts with 1 at the begining of the movement and converges exponentially to 0.
#     #     """
#     #     phases = torch.exp(-self.alpha/3. * torch.linspace(0, 1, n_steps))
#     #     if t is None:
#     #         return phases
#     #     else:
#     #         return phases[t]

#     def force(self, zs, g):
#         """Network to generate force for each frame
#         Parameters:
#             zs (tensor) -- list of latent codes [len_sequence, batch, z_dim]
#         Returns:
#             f (tensor) -- force of current frame [len_sequence, batch, z_dim]
#         """
#         zn = zs[-1]
#         dz = self.d(zs)
#         ddz = self.dd(zs)

#         f = self.tau*self.tau*ddz - self.alpha*(
#             self.beta*(zn - zs) - self.tau*dz)

#         return f

#     def d(self, x):
#         d = torch.zeros_like(x)
#         if x.shape[0] == 1:
#             return d
#         else:
#             for i in range(x.shape[0]):
#                 if i == 0:
#                     d[i,...] = x[i+1,...] - x[i,...]
#                 elif i == x.shape[0]-1:
#                     d[i,...] = x[i,...] - x[i-1,...]
#                 else:
#                     d[i,...] = x[i+1,...] - x[i-1,...] / 2.
#             return d

#     def dd(self, x):
#         return self.d(self.d(x))


#     def forward(self, x):
#         """Calculate VAE-DMP's output. Take input as sequence.
#         Parameters:
#             x (tensor) - - observation inputs of shape [batch, len_sequence, x_dim]
#         Returns:
#             Latent codes zs and reconstructed xs.
#         """
#         # first, reshape x from [batch, len_sequence, x_dim] to [len_sequence, batch, x_dim]
#         x = torch.transpose(x, 0, 1)
#         # z1
#         feature1 = self.encoder(x[0])
#         wd1, w1, _ = self.init_enc(feature1)
#         z1 = feature1
#         zz1 = z1.clone()
#         x1 = self.decoder(z1)
#         # z2
#         # _, _, z2 = self.init_enc(features[1])
#         feature2 = self.encoder(x[1])
#         z2 = feature2
#         # z_goal
#         # _, _, zn = self.init_enc(features[-1])
#         featuren = self.encoder(x[-1])
#         zn = featuren

#         dz1 = (z2 - z1) / self.dt
#         zt = z1.clone()
#         dzt = dz1.clone()

#         xs = x1.unsqueeze(0)
#         zs = z1.unsqueeze(0)
#         zzs = zz1.unsqueeze(0)
#         wds = wd1.unsqueeze(0)
#         ws = w1.unsqueeze(0)
#         for xt in x[1:]:
#             # calculate f
#             fs = self.force(zs, zn)
#             ft = fs[-1]
#             featuret = self.encoder(xt)
#             wdt, wt = self.noise_net(featuret, zt, ft)
#             wds = torch.cat((wds, wdt.unsqueeze(0)), 0)
#             # calculate z[t+1]
#             nex = self.A(zt, dzt) + self.b(zn, ft, wt)
#             zt = nex[0]
#             dzt = nex[1]
#             zs = torch.cat((zs, zt.unsqueeze(0)), 0)
#             # calculate zz[t+1]
#             zzt = featuret
#             zzs = torch.cat((zzs, zzt.unsqueeze(0)), 0)
#             # calculate reconstructed x
#             xt = self.decoder(zt)
#             xs = torch.cat((xs, xt.unsqueeze(0)), 0)

#         xs = torch.transpose(xs, 0, 1)
#         zs = torch.transpose(zs, 0, 1)
#         zzs = torch.transpose(zzs, 0, 1)
#         wds = torch.transpose(wds, 0, 1)
#         fs = torch.transpose(fs, 0, 1)
#         return xs, zs, zzs, wds, fs



class VAEDMPLoss(nn.Module):
    def __init__(self):
        super(VAEDMPLoss, self).__init__()


    def __call__(self, wd, outputs, inputs):
        dim = int(wd.shape[-1]/2)
        w_mean, w_logstd = torch.split(wd, [dim, dim], dim=-1)
        # w_std = torch.exp(w_logstd) + 1e-3
        # w_dist = torch.distributions.normal.Normal(loc=w_mean, scale=w_std)
        # prior_dist = torch.distributions.normal.Normal(loc=torch.zeros_like(w_mean), scale=torch.ones_like(w_std))
        # kl_loss = F.kl_div(w_dist.sample(), prior_dist.sample(), reduction = 'sum')
        kl_loss = torch.sum(1. + w_logstd - w_mean.pow(2) - w_logstd.exp())
        kl_loss *= -0.5

        #inputs = torch.transpose(inputs, 0, 1)
        recons_loss = F.mse_loss(outputs, inputs, reduction = 'sum')
        #recons_loss *= 0.5

        loss = recons_loss + kl_loss
        #print(inputs, '----', outputs)
        return loss

class VAEDMPForceLoss(nn.Module):
    def __init__(self):
        super(VAEDMPForceLoss, self).__init__()

    def __call__(self, fs):
        #delta force term
        delta_fs = fs[:,1:,...] - fs[:,:-1,...]
        delta_fs = torch.linalg.norm(delta_fs, dim=-1)
        max_delta_fs = torch.max(delta_fs, 1).values
        min_delta_fs = torch.min(delta_fs, 1).values
        delta_delta_fs = max_delta_fs - min_delta_fs
        target = torch.zeros_like(delta_delta_fs)
        delta_fs_loss = F.mse_loss(delta_delta_fs, target, reduction = 'sum')

        return delta_fs_loss


class VAEDMPZLoss(nn.Module):
    def __init__(self):
        super(VAEDMPZLoss, self).__init__()

    def __call__(self, z, zz):
        #delta force term
        diff = z - zz
        target = torch.zeros_like(diff)
        z_loss = F.mse_loss(diff, target, reduction = 'sum')

        return z_loss

################################################################################
#                                                                              #
#                              LSTM implementation                             #
#                                                                              #
################################################################################
# class LSTM(nn.Module):
#     def __init__(self, input_size, fo_size, hidden_size, batch_first=True):
#         super(LSTM, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.fo_size = fo_size
#         self.num_layers = 1
#         self.batch_first = batch_first

#         self.input_weights = nn.Linear(input_size, 4 * hidden_size)
#         self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)
#         self.fo_weights = nn.Linear(fo_size, 4 * hidden_size)

#     def forward(self, x, fo, hidden, ctx, ctx_mask=None):
#         def recurrence(x, fo, hidden):
#             hx, cx = hidden
#             gates = self.input_weights(x) + self.hidden_weights(hx) + self.fo_weights(fo)
#             ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

#             ingate = F.sigmoid(ingate)
#             forgetgate = F.sigmoid(forgetgate)
#             cellgate = F.tanh(cellgate)
#             outgate = F.sigmoid(outgate)

#             cy = (forgetgate * cx) + (ingate * cellgate)
#             hy = outgate * F.tanh(cy)

#             return hy, cy

#         if self.batch_first:
#             x = x.transpose(0, 1)

#         output = []
#         steps = range(x.size(0))
#         for i in steps:
#             hidden = recurrence(x[i], fo[i], hidden)
#             if isinstance(hidden, tuple):
#                 output.append(hidden[0])
#             else:
#                 output.append(hidden)

#         output = torch.cat(output, 0).view(input.size(0), *output[0].size())

#         if self.batch_first:
#             output = output.transpose(0, 1)

#         return output, hidden


class LSTMCell(nn.Module):
    def __init__(self, input_size, fo_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.fo_size = fo_size
        self.hidden_size = hidden_size
        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)
        self.fo_weights = nn.Linear(fo_size, 4 * hidden_size)

    def forward(self, input, fo, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        gates = self.input_weights(input) + self.hidden_weights(hx) + self.fo_weights(fo)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy


# define RTNCL algorithm
class RTNCL(nn.Module):
    def __init__(self, x_dim, t_dim, hidden_dim, is_decoder, o_dim=0, p_dim=0):
        super(RTNCL, self).__init__()

        self.x_dim = x_dim
        self.p_dim = p_dim
        self.t_dim = t_dim
        self.o_dim = o_dim
        self.to_dim = t_dim + o_dim
        self.hidden_dim = hidden_dim

        # encoder
        self.fc_en1 = nn.Linear(self.x_dim+self.p_dim, 512)
        self.fc_en2 = nn.Linear(512, 4*self.hidden_dim)

        # global offset encoder
        if self.o_dim != 0:
            self.fc_o_en1 = nn.Linear(self.o_dim, 128)
            self.fc_o_en2 = nn.Linear(128, 4*self.hidden_dim)

        # target encoder
        self.fc_f_en1 = nn.Linear(self.t_dim, 128)
        self.fc_f_en2 = nn.Linear(128, 4*self.hidden_dim)

        # initial network
        self.fc_init_en1 = nn.Linear(self.x_dim, 512)
        self.fc_init_en2 = nn.Linear(512, 4*self.hidden_dim)

        # transition
        self.lstmCell = LSTMCell(4*self.hidden_dim, 4*self.hidden_dim, 4*self.hidden_dim)


        # decoder
        self.fc_de1 = nn.Linear(4*self.hidden_dim, 256)
        self.fc_de2 = nn.Linear(256, 128)
        self.fc_de3 = nn.Linear(128, self.x_dim)


    def encoder(self, x, p=None):
        """Encoder network to extract features from inputs X
        Parameters:
            x (tensor) -- inputs in observation space of shape [batch, len_sequence, x_dim]
            p (tensor) -- local terrain patch
        """
        if p is not None:
            x = torch.cat((x, p), dim = -1)

        h1 = F.leaky_relu(self.fc_en1(x))
        h2 = F.leaky_relu(self.fc_en2(h1))
        return h2

    def decoder(self, h):
        """Decoder network to recover x from h
        Parameters:
            h (tensor) -- hidden states of shape [batch, len_sequence, z_dim]
        """
        h1 = F.leaky_relu(self.fc_de1(h))
        h2 = F.leaky_relu(self.fc_de2(h1))
        h3 = F.leaky_relu(self.fc_de3(h2))
        return h3

    def f_net(self, t):
        """Encoder network to embed target information
        Parameters:
            t (tensor) -- target information of shape [batch, len_sequence, f_dim]
        """
        h1 = F.leaky_relu(self.fc_f_en1(t))
        h2 = F.leaky_relu(self.fc_f_en2(h1))
        return h2

    def o_net(self, o):
        """Encoder network to embed offset information
        Parameters:
            o (tensor) -- offset information of shape [batch, len_sequence, o_dim]
        """
        h1 = F.leaky_relu(self.fc_o_en1(o))
        h2 = F.leaky_relu(self.fc_o_en2(h1))
        return h2

    def h_init_net(self, x):
        """MLP to predict initial hidden state h0
        Parameters:
        x (tensor) -- The first frame from input sequence of shape [batch, x_dim]
        """
        h1 = F.leaky_relu(self.fc_init_en1(x))
        h2 = F.leaky_relu(self.fc_init_en2(h1))
        return h2, h2


    def forward(self, xs, ts, os=None, ps=None):
        """Calculate RTNCL's output.
        Parameters:
            xs (tensor) -- observation inputs of shape [batch, len_sequence, x_dim]
            ts (tensor) -- target frames (T and T+1) as conditioning information
            os (tensor) -- global offsets from target
            ps (tensor) -- local terrain patch
        Returns:
            hs (tensor) -- transition hidden states
            xs (tensor) -- transition frames
        """

        # conver to [len_sequence, batch, dim]
        xs = torch.transpose(xs, 0, 1)
        if os is not None:
            os = torch.transpose(os, 0, 1)
        if ps is not None:
            ps = torch.transpose(ps, 0, 1)

        hs = []
        xs_next = []
        ht, ct = self.h_init_net(xs[0])
        for i in range(xs.shape[0]):
            he = self.encoder(xs[i])
            hf = self.f_net(ts)
            if os is not None:
                ho = self.o_net(os[i])
                hfo = torch.cat((hf, ho), dim = -1)
            else:
                hfo = hf

            ht, ct = self.lstmCell(he, hfo, (ht, ct))
            h = self.decoder(ht)
            hs.append(h)
            x_next = x + h
            xs_next.append(x_next)

        for i in range(self.total_len - self.past_len - 2):
            he = self.encoder(x_next)
            hf = self.f_net(ts)
            if os is not None:
                # Not implemented at this moment!
                ho = self.o_net(os[i])
                hfo = torch.cat((hf, ho), dim = -1)
            else:
                hfo = hf

            ht, ct = self.lstmCell(he, (ht, ct))
            h = self.decoder(ht)
            hs.append(h)
            x_next = x_next + h
            xs_next.append(x_next)


        hs = torch.stack(hs)
        xs_next = torch.stack(xs_next)
        hs = torch.transpose(hs, 0, 1)
        xs_next = torch.transpose(xs_next, 0, 1)

        return hs, xs_next


class RTNCLLoss(nn.Module):
    def __init__(self):
        super(RTNCLLoss, self).__init__()

    def __call__(self, outputs, inputs):
        loss = 1000*F.mse_loss(outputs, inputs, reduction = 'sum')
        return loss
        


class RTN(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim, is_decoder, transition_len=19, past_len=10, target_len=1):
        super(RTN, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        self.past_len = past_len
        self.target_len = target_len
        self.transition_len = transition_len

        self.is_decoder = is_decoder
        # encoder
        # self.fc_en1 = nn.Linear(self.x_dim, self.hidden_dim)
        # self.fc_en2 = nn.Linear(self.hidden_dim, 4*self.z_dim)


        # initial network
        self.fc_init_en1 = nn.Linear(self.x_dim, self.hidden_dim)
        self.fc_init_en2 = nn.Linear(self.hidden_dim, 4*self.z_dim)

        # transition
        self.lstm = nn.LSTM(self.x_dim, 4*self.z_dim)


        # decoder
        self.fc_de1 = nn.Linear(4*self.z_dim, self.hidden_dim//2)
        self.fc_de2 = nn.Linear(self.hidden_dim//2, self.hidden_dim//4)
        self.fc_de3 = nn.Linear(self.hidden_dim//4, self.x_dim)


    # def encoder(self, x):
    #     """Encoder network to extract features from inputs X
    #     Parameters:
    #         x (tensor) -- inputs in observation space of shape [batch, len_sequence, x_dim]
    #     """

    #     h1 = F.leaky_relu(self.fc_en1(x))
    #     h2 = F.leaky_relu(self.fc_en2(h1))
    #     return h2

    def decoder(self, h):
        """Decoder network to recover x from h
        Parameters:
            h (tensor) -- hidden states of shape [batch, len_sequence, z_dim]
        """
        h1 = F.leaky_relu(self.fc_de1(h))
        h2 = F.leaky_relu(self.fc_de2(h1))
        h3 = F.leaky_relu(self.fc_de3(h2))
        return torch.sigmoid(h3)


    def h_init_net(self, x):
        """MLP to predict initial hidden state h0
        Parameters:
        x (tensor) -- The first frame from input sequence of shape [batch, x_dim]
        """
        h1 = F.leaky_relu(self.fc_init_en1(x))
        h2 = F.leaky_relu(self.fc_init_en2(h1))
        return h2, h2


    def force(self, z, g):
        """Network to generate force for each frame
        Parameters:
            z (tensor) -- list of latent codes [len_sequence, batch, z_dim]
        Returns:
            f (tensor) -- force of current frame [len_sequence, batch, z_dim]
        """
        alpha=25.0
        beta=25.0/4.0
        tau=1.0

        dz = self.d(z)
        ddz = self.dd(z)

        f = tau*tau*ddz - alpha*(
            beta*(g - z) - tau*dz)
        f = torch.sigmoid(f)

        return f

    def d(self, x):
        d = torch.zeros_like(x)
        if x.shape[0] == 1:
            return d
        else:
            for i in range(x.shape[0]):
                if i == 0:
                    d[i,...] = x[i+1,...] - x[i,...]
                elif i == x.shape[0]-1:
                    d[i,...] = x[i,...] - x[i-1,...]
                else:
                    d[i,...] = x[i+1,...] - x[i-1,...] / 2.
            return d

    def dd(self, x):
        return self.d(self.d(x))


    # def slerp(p0, p1, t):
    #     """
    #     Spherical linear interpolation
    #     """
    #     omega = torch.acos(torch.dot(torch.squeeze(p0/torch.linalg.norm(p0)),
    #                              torch.squeeze(p1/torch.linalg.norm(p1))))
    #     so = torch.sin(omega)
    #     return torch.sin(1.0 - t) * omega / so * p0 + torch.sin(t * omega) / so * p1

    # def blend(outputs, target):
    #     """Calculate interpolation between choosen outputs and target
    #     Parameters:
    #         outputs (tensor) -- tensor of latent codes [len_sequence, batch, z_dim]
    #         target (tensor) -- tensor of latent codes [len_sequence, batch, z_dim]
    #     Returns:
    #         res (tensor) -- transition frames [len_sequence, batch, z_dim]
    #     """
    #     # find output with least L2 distance to target
    #     dists_o2t = outputs - target
    #     dists_o2o = outputs[1:,...] - outputs[:-1,...]
    #     dists_o2t = torch.linalg.norm(dists_o2t, dim=-1)
    #     dists_o2o = torch.linalg.norm(dists_o2o, dim=-1)
    #     dists_o2o_max, _ = torch.max(dists_o2o, dim=0)

    #     dists_o2t_min, o2t_minidx = torch.min(dists_o2t, dim=0)

    #     need_blend = dists_o2t_min > dists_o2o_max
    #     n_steps = torch.ceil(dists_o2t_min / dists_o2o_max)

    def forward(self, xs, isTrain):
        """Calculate RTN's output.
        Parameters:
            xs (tensor) -- observation inputs of shape [batch, len_sequence, x_dim]
        Returns:
            hs (tensor) -- transition hidden states
            xs (tensor) -- transition frames
        """

        # conver to [len_sequence, batch, dim]
        xs = torch.transpose(xs, 0, 1)

        hs = []
        xs_next = []
        fs = []
        ht, ct = self.h_init_net(xs[0])
        ht = torch.unsqueeze(ht, 0)
        ct = torch.unsqueeze(ct, 0)
        last_x = xs[0:self.past_len]
        for t in range(self.transition_len):
            if isTrain:
                x = xs[t:t+self.past_len]
            else:
                if t == 0:
                    x = last_x
                else:
                    x = last_x[1:]
                    x = torch.cat((x,torch.unsqueeze(x_next, 0)),0)


            out, (ht, ct) = self.lstm(x, (ht, ct))
            out = out[-1]
            h = self.decoder(out)
            hs.append(h)
            #x_next = x[-1] + h
            x_next = h
            xs_next.append(x_next)

            last_x = x

            #calculate f
            f = self.force(torch.stack(xs_next), xs[-1])


        hs = torch.stack(hs)
        xs_next = torch.stack(xs_next)
        hs = torch.transpose(hs, 0, 1)
        xs_next = torch.transpose(xs_next, 0, 1)
        f = torch.transpose(f, 0, 1)

        return xs_next, hs, f

class RTNRecLoss(nn.Module):
    def __init__(self):
        super(RTNRecLoss, self).__init__()

    def __call__(self, outputs, targets):
        loss = 100*F.mse_loss(outputs, targets, reduction = 'sum')
        return loss

class RTNMonoLoss(nn.Module):
    def __init__(self):
        super(RTNMonoLoss, self).__init__()

    def __call__(self, outputs):
        """
        Parameters:
            outputs (tensor) -- force term at each time step of shape [batch, len_sequence, f_dim]
        Returns:
            loss (float) -- loss
        """
        f_norm = torch.linalg.norm(outputs, dim=-1)
        f_norm_delta = f_norm[:,1:] - f_norm[:,:-1]

        # instead of using ones, we use a positive number to represent how far it goes away
        lossmap = torch.zeros_like(f_norm_delta)
        targets = torch.zeros_like(f_norm_delta)

        mask = f_norm_delta > 0
        lossmap[mask] = f_norm_delta[mask]

        loss = F.mse_loss(lossmap, targets, reduction = 'sum')
        return loss



        