import os
import argparse


class BaseOptions(object):

    def __init__(self):
        self.initialized = False
        self.parser = None
        self.opt = None
        # config for predefined method
        self.override_dict = {
            'InsDis': ['RGB', False, 'bank', 'A', 'linear', 0.07],
            'CMC': ['CMC', False, 'bank', 'C', 'linear', 0.07],
            'MoCo': ['RGB', False, 'moco', 'A', 'linear', 0.07],
            'PIRL': ['RGB', True, 'bank', 'A', 'linear', 0.07],
            'MoCov2': ['RGB', False, 'moco', 'B', 'mlp', 0.2],
            'CMCv2': ['CMC', False, 'moco', 'E', 'mlp', 0.2],
            'InfoMin': ['RGB', True, 'moco', 'D', 'mlp', 0.15],
        }

    def initialize(self, parser):
        # specify folder
        parser.add_argument('--image_folder', type=str, default='AIGC-3K/image',
                            help='path to data')
        parser.add_argument('--csv_file', type=str, default="AIGC-3K/data.csv",
                            help='path to csv file')
        parser.add_argument('--checkpoints', type=str, default="exp",
                            help='path to save model')
        parser.add_argument('--tb_path', type=str, default="runs",
                            help='path to tensorboard')
        parser.add_argument('--aigciqa_root_path', type=str, default="AIGCIQA2023",
                            help='AIGCIQA2023 dataset path')

        # basics
        parser.add_argument('--print_freq', type=int, default=10,
                            help='print frequency')
        parser.add_argument('--save_freq', type=int, default=30,
                            help='save frequency')
        parser.add_argument('--batch_size', type=int, default=16,
                            help='batch_size')
        parser.add_argument('-j', '--num_workers', type=int, default=0,
                            help='num of workers to use')
        parser.add_argument('-n_scale', '--n_scale', type=int, default=1,
                            help='num of scales per image to use. 1 only image, 2 image and half resized image')
        parser.add_argument('-n_distortions', '--n_distortions', type=int, default=1,
                            help='num of distortions per image crop to use. 1 for single distortion image, 2 for randomly selecting amoing 1/2 distortions')
        parser.add_argument('-patch_size', '--patch_size', type=int, default=224,
                            help='patch_size to crop for each image')
        parser.add_argument('-swap_crops', '--swap_crops', type=int, default=1,
                            help='patch_size to crop for each image')
        parser.add_argument('-random_seed', type=int, default=2023,
                            help='random_seed')
        parser.add_argument('--is_training', action='store_true', help='training mode')
        parser.add_argument('--itr', type=int, default=1, help='itr')
        parser.add_argument('--tolerance', type=int, default=4, help='tolerance')

        # optimization
        parser.add_argument('--epochs', type=int, default=20,
                            help='number of training epochs')
        parser.add_argument('--learning_rate', type=float, default=0.05,
                            help='learning rate')
        parser.add_argument('--lr_decay_epochs', type=str, default='120,160',
                            help='where to decay lr, can be a list')
        parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                            help='decay rate for learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-4,
                            help='weight decay')
        parser.add_argument('--momentum', type=float, default=0.9,
                            help='momentum for SGD')
        parser.add_argument('--cosine', action='store_true',
                            help='using cosine annealing')
        parser.add_argument('--optimizer', type=str, default='SGD',
                            help='SGD/AdamW/LARS')
        parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

        # method selection
        parser.add_argument('--method', default='Customize', type=str,
                            choices=['InsDis', 'CMC', 'CMCv2', 'MoCo', 'MoCov2',
                                     'PIRL', 'InfoMin', 'Customize'],
                            help='Choose predefined method. Configs will be override '
                                 'for all methods except for `Customize`, which allows '
                                 'for user-defined combination of methods')
        # method configuration
        parser.add_argument('--modal', default='RGB', type=str, choices=['RGB', 'CMC'],
                            help='single RGB modal, or two modalities in CMC')
        parser.add_argument('--jigsaw', action='store_true',
                            help='adding PIRL branch')
        parser.add_argument('--mem', default='bank', type=str, choices=['bank', 'moco'],
                            help='memory mechanism: memory bank, or moco encoder cache')

        # model setup
        parser.add_argument('--arch', default='resnet50', type=str,
                            help='e.g., resnet50, resnext50, resnext101'
                                 'and their wider variants, resnet50x4')
        parser.add_argument('-d', '--feat_dim', default=128, type=int,
                            help='feature dimension for contrastive loss')
        parser.add_argument('-k', '--nce_k', default=65536, type=int,
                            help='number of negatives')
        parser.add_argument('-m', '--nce_m', default=0.5, type=float,
                            help='momentum for memory update')
        parser.add_argument('-t', '--nce_t', default=0.07, type=float,
                            help='temperature')
        parser.add_argument('--alpha', default=0.999, type=float,
                            help='momentum coefficients for moco encoder update')
        parser.add_argument('--head', default='linear', type=str,
                            choices=['linear', 'mlp'], help='projection head')

        # resume
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--start_epoch', default=0, type=int)

        # gpu_training
        parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        parser.add_argument('--gpu', type=int, default=0, help='gpu')
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

        return parser

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>35}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        # print(message)

    def modify_options(self, opt):
        raise NotImplementedError

    def override_options(self, opt):
        # override parameters for predefined method
        if opt.method in self.override_dict.keys():
            opt.modal = self.override_dict[opt.method][0]
            opt.jigsaw = self.override_dict[opt.method][1]
            opt.mem = self.override_dict[opt.method][2]
            opt.aug = self.override_dict[opt.method][3]
            opt.head = self.override_dict[opt.method][4]
            opt.nce_t = self.override_dict[opt.method][5]
        return opt

    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser('arguments options')
            parser = self.initialize(parser)
            self.parser = parser
            self.initialized = True
        else:
            parser = self.parser

        opt = parser.parse_args()
        opt = self.modify_options(opt)
        self.opt = opt

        # self.print_options(opt)

        return opt
