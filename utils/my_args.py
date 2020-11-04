import torch
from torch.utils.data import DataLoader
from data.config import voc, helmet, MEANS
from data.helmet import HELMET_ROOT, HelmetDetection
from data.voc0712 import VOC_ROOT, VOCDetection
from utils.augmentations import SSDAugmentation
from data import detection_collate
import argparse


class Arguments:
    def __init__(self):
        self.dataset = 'VOC'
        self.cuda = True
        self.resume_pth = 'weights/big_net_VOC.pth'
        self.batch_size = 32
        self.num_workers = 4
        self.cfg = None

    def init_dataset(self, init_dataloader=True):
        if self.dataset == 'COCO':
            raise NotImplementedError()
        elif self.dataset == 'VOC':
            # raise NotImplementedError()
            self.cfg = voc
            dataset = VOCDetection(root=VOC_ROOT, transform=SSDAugmentation(self.cfg['min_dim'], MEANS))
        elif self.dataset == 'helmet':
            self.cfg = helmet
            dataset = HelmetDetection(root=HELMET_ROOT, transform=SSDAugmentation(self.cfg['min_dim'], MEANS))
        else:
            raise RuntimeError()
        if not init_dataloader:
            return dataset
        data_loader = DataLoader(dataset, self.batch_size,
                                 num_workers=self.num_workers,
                                 shuffle=True, collate_fn=detection_collate,
                                 pin_memory=True)
        return dataset, data_loader

    def try_enable_cuda(self):
        if torch.cuda.is_available():
            if self.cuda:
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            if not self.cuda:
                print("WARNING: It looks like you have a CUDA device, but aren't " +
                      "using CUDA.\nRun with --cuda for optimal training speed.")
                torch.set_default_tensor_type('torch.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')


class TrainASDArguments:
    def __init__(self):
        self.dataset = 'VOC'
        self.cuda = True
        self.resume_pth = 'weights/big_net_VOC.pth'
        self.batch_size = 32
        self.num_workers = 4
        self.cfg = None

        def str2bool(v):
            return v.lower() in ("yes", "true", "t", "1")
        parser = argparse.ArgumentParser(
            description='Auto-tailored Small Detector Training With Pytorch')
        train_set = parser.add_mutually_exclusive_group()
        parser.add_argument('--dataset', default='VOC',
                            type=str, help='VOC, COCO, etc.')
        parser.add_argument('--basenet', default=None,
                            help='Pretrained base model')
        parser.add_argument('--batch_size', default=32, type=int,
                            help='Batch size for training')
        parser.add_argument('--resume', default=None, type=str,
                            help='Checkpoint state_dict file to resume training from')
        parser.add_argument('--start_iter', default=0, type=int,
                            help='Resume training at this iter')
        parser.add_argument('--num_workers', default=4, type=int,
                            help='Number of workers used in dataloading')
        parser.add_argument('--cuda', default=True, type=str2bool,
                            help='Use CUDA to train model')
        parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                            help='initial learning rate')
        parser.add_argument('--momentum', default=0.9, type=float,
                            help='Momentum value for optim')
        parser.add_argument('--weight_decay', default=5e-4, type=float,
                            help='Weight decay for SGD')
        parser.add_argument('--gamma', default=0.1, type=float,
                            help='Gamma update for SGD')
        parser.add_argument('--save_folder', default='weights/',
                            help='Directory for saving checkpoint models')
        args = parser.parse_args()

        if torch.cuda.is_available():
            if args.cuda:
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            if not args.cuda:
                print("WARNING: It looks like you have a CUDA device, but aren't " +
                      "using CUDA.\nRun with --cuda for optimal training speed.")
                torch.set_default_tensor_type('torch.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        if not os.path.exists(args.save_folder):
            os.mkdir(args.save_folder)

    def init_dataset(self, init_dataloader=True):
        if self.dataset == 'COCO':
            raise NotImplementedError()
        elif self.dataset == 'VOC':
            # raise NotImplementedError()
            self.cfg = voc
            dataset = VOCDetection(root=VOC_ROOT, transform=SSDAugmentation(self.cfg['min_dim'], MEANS))
        elif self.dataset == 'helmet':
            self.cfg = helmet
            dataset = HelmetDetection(root=HELMET_ROOT, transform=SSDAugmentation(self.cfg['min_dim'], MEANS))
        else:
            raise RuntimeError()
        if not init_dataloader:
            return dataset
        data_loader = DataLoader(dataset, self.batch_size,
                                 num_workers=self.num_workers,
                                 shuffle=True, collate_fn=detection_collate,
                                 pin_memory=True)
        return dataset, data_loader

    def try_enable_cuda(self):
        if torch.cuda.is_available():
            if self.cuda:
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            if not self.cuda:
                print("WARNING: It looks like you have a CUDA device, but aren't " +
                      "using CUDA.\nRun with --cuda for optimal training speed.")
                torch.set_default_tensor_type('torch.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')