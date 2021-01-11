import argparse
import os
from utils import utils
import torch
import models
import data


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        """This class defines options used during both training and test time.

        It also implements several helper functions such as parsing, printing, and saving the options.
        It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
        """
        parser.add_argument('--dataroot', type=str, help='path to data')
        parser.add_argument('--randomroot', type=str, help='path to random data')
        parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
        parser.add_argument('--input_latent', type=int, default=512, help='size of single input latant data')
        parser.add_argument('--output_latent', type=int, default=512, help='size of single output latent data')
        parser.add_argument('--z_size', type=int, default=0, help='#latent vector')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')
        parser.add_argument('--name', type=str, default='', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--dataset_mode', type=str, default='aligned_path', help='aligned_path, random_path')
        parser.add_argument('--model', type=str, default='path_gan', help='chooses which model to use. bicycle,, ...')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--epoch2', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--num_threads', default=4, type=int, help='# sthreads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='../results', help='models are saved here')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes inputs in order to make batches, otherwise takes them randomly')
        parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--a_mode', type=str, default='ones',help='zeros|ones|linear|geodesic; the method to fill in the intermediate frames between head and tail')
        parser.add_argument('--path_length', type=int, default=10, help='the length of the path')
        parser.add_argument('--key_frames', '--list', nargs='+', type=int, help='the key frames in one path')
        parser.add_argument('--is_decoder', action='store_true', help='use for decoder')

        # model parameters
        parser.add_argument('--norm', type=str, default='none', help='instance normalization or batch normalization')
        parser.add_argument('--nl', type=str, default='relu', help='non-linearity activation: relu | lrelu | elu')
        parser.add_argument('--num_downs', type=int, default=7, help='# of downsamplings in Generator')
        parser.add_argument('--d_layers', type=int, default=3, help='# of layers in Discrminators')
        parser.add_argument('--dim_heatmap', type=int, default=64, help='the dimension of the 3D heatmap, the heatmap has to be a cube')
        parser.add_argument('--sigma', type=float, default=0.05, help='sigma of the 3D heatmap')
        parser.add_argument('--z_dim', type=int, default=512, help='dimension of the latent space')
        parser.add_argument('--pca_dim', type=int, default=2048, help='dimension of the pca inside the VAE')
        parser.add_argument('--num_joints', type = int, default=17, help='number of joints')

        # VAEDMP parameters
        parser.add_argument('--x_dim', type=int, default=72, help='dimension of single pose input')
        parser.add_argument('--hidden_dim', type=int, default=512, help='dimension of hidden layers')
        parser.add_argument('--transform_dim', type=int, default=128, help='dimension of transform layers')
        parser.add_argument('--noise_dim', type=int, default=32, help='dimension of noise layers')
        parser.add_argument('--u_dim', type=int, default=32, help='dimension of input u')

        # extra parameters
        parser.add_argument('--where_add', type=str, default='none', help='input|all|middle; where to add z in the network G')
        parser.add_argument('--conditional_D', action='store_true', help='if use conditional GAN for D')
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=1., help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size')

        # Lafan dataset parameters
        parser.add_argument('--lafan_is_quat', action='store_true', help='quaternion or position')
        parser.add_argument('--lafan_mode', type=str, default='pose', help='choose pose or sequence')
        parser.add_argument('--lafan_window', type=int, default=30, help='length of sequence')
        parser.add_argument('--lafan_offset', type=int, default=20, help='offset to sample the data')
        parser.add_argument('--lafan_samplerate', type=int, default=5, help='sample rate')
        parser.add_argument('--lafan_minmax_path', type=str, default='None', help='minmax file path')

        # H36M dataset parameters
        parser.add_argument('--h36m_mode', type=str, default='pose', help='choose pose or sequence')

        # special tasks
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are difined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        utils.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
