import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect
import ssl

# from __future__ import print_function

# import torch
# import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn

import transforms as transforms
from dataloader import lunanod
import os
# import argparse
# import time
# from utils import progress_bar
# from torch.autograd import Variable
import logging
# import numpy as np
import ast
import pandas as pd

from separate_loss import ConvSeparateLoss, TriSeparateLoss

# from utils_mamba.lr_scheduler import build_scheduler
# from utils_mamba.optimizer import build_optimizer
# from config import get_config

# mamba
import os
import time
import json
import random
import argparse
import datetime
import tqdm
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config_2 import get_config
# from models import build_model
# from data import build_loader
from utils_mamba.lr_scheduler import build_scheduler
from utils_mamba.optimizer import build_optimizer
from utils_mamba.logger import create_logger
from utils_mamba.utils import  NativeScalerWithGradNormCount, auto_resume_helper, reduce_tensor
from utils_mamba.utils import load_checkpoint_ema, load_pretrained_ema, save_checkpoint_ema
from repro_utils import capture_rng_state, cycling_batches, restore_rng_state, seed_everything

from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count

from timm.utils import ModelEma as ModelEma

# if torch.multiprocessing.get_start_method() != "spawn":
#     print(f"||{torch.multiprocessing.get_start_method()}||", end="")
#     torch.multiprocessing.set_start_method("spawn", force=True)

# torch._dynamo.config.cache_size_limit = 1024
parser = argparse.ArgumentParser("cifar")
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--crop_dir', type=str, required=True,
                    help='directory containing the preprocessed 32^3 nodule .npy crops')
parser.add_argument('--luna_root', type=str, required=True,
                    help='directory containing LUNA16 subset0 through subset9 directories')
parser.add_argument('--annotations_path', type=str,
                    default=os.path.join(_REPO_ROOT, 'data', 'annotationdetclsconvfnl_v3.csv'),
                    help='CSV containing seriesuid, coordinates, diameter, and malignancy labels')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')

# Original search space
# parser.add_argument('--batch_size', type=int, default=22, help='batch size')
# new search space
parser.add_argument('--batch_size', type=int, default=4, help='batch size')

parser.add_argument('--learning_rate', type=float, default=2e-4,
                    help='published base learning rate (mirrored by config_2.py)')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
# parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
# parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')

parser.add_argument('--layers', type=int, default=8, help='total number of layers')
# parser.add_argument('--layers', type=int, default=4, help='total number of layers')
# parser.add_argument('--layers', type=int, default=8, help='total number of layers')

parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
# 简单
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
# 复杂
# parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
# loss权
parser.add_argument('--aux_loss_weight', type=float, default=10.0, help='weight decay')
# parser.add_argument('--aux_loss_weight', type=float, default=20.0, help='weight decay')
# parser.add_argument('--aux_loss_weight', type=float, default=100.0, help='weight decay')


parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--arch_subset', type=int, default=0, choices=range(10),
                    help='LUNA16 subset used only for architecture-parameter updates; the paper uses subset0')
parser.add_argument('--fold', type=int, default=None,
                    help='deprecated compatibility option; only --fold 1 matches the published subset0 protocol')
parser.add_argument('--resume', type=str, default='', help='search checkpoint to resume')
# parser.add_argument('--fold', type=int, default=8, help="fold")
# Original search space
# parser.add_argument('--pret', type=int, default=18, help="pret")
# new search space
# parser.add_argument('--pret', type=int, default=20, help="pret")
parser.add_argument('--pret', type=int, default=60, help="pret")
# parser.add_argument('--pret', type=int, default=-1, help="pret")
# parser.add_argument('--pret', type=int, default=70, help="pret")
# parser.add_argument('--pret', type=int, default=30, help="pret")
# parser.add_argument('--pret', type=int, default=250, help="pret")
# parser.add_argument('--pret', type=int, default=110, help="pret")

# 就算换加权依然富集maxpool，还是开始跑的快虽然不影响别的操作但是已经跑远了
# parser.add_argument('--pret', type=int, default=0, help="pret")
# fair-darts
parser.add_argument('--sep_loss', type=str, default='l2', help='path to save the model')
# parser.add_argument('--sep_loss', type=str, default='l1', help='path to save the model')




# parser.add_argument('--cfg', type=str, metavar="FILE",
#                     default="./configs/wms/vmambav_wms_3.yaml",
#                     help='path to config file', )

# parser.add_argument('--cfg', type=str, metavar="FILE",
#                     default="",
#                     help='path to config file', )

# parser.add_argument('--cfg', type=str, metavar="FILE",
#                     default="",
#                     help='path to config file', )


# parser.add_argument('--optim', type=str, help='overwrite optimizer if provided, can be adamw/sgd.')


args = parser.parse_args()

if args.fold is not None:
    if args.fold != 1 or args.arch_subset != 0:
        parser.error('the published protocol requires subset0 only; use --arch_subset 0 (legacy --fold 1)')
    logging.warning('--fold is deprecated; use --arch_subset 0')

for option, path in (('--crop_dir', args.crop_dir), ('--luna_root', args.luna_root)):
    if not os.path.isdir(path):
        parser.error('{} must be an existing directory: {}'.format(option, path))
if not os.path.isfile(args.annotations_path):
    parser.error('--annotations_path must be an existing file: {}'.format(args.annotations_path))
if args.resume and not os.path.isfile(args.resume):
    parser.error('--resume must be an existing checkpoint: {}'.format(args.resume))

config = get_config(args)

CROPSIZE = 32
gbtdepth = 1
arch_subset = args.arch_subset
pret = args.pret
blklst = []  # ['1.3.6.1.4.1.14519.5.2.1.6279.6001.121993590721161347818774929286-388', \
# '1.3.6.1.4.1.14519.5.2.1.6279.6001.121993590721161347818774929286-389', \
# '1.3.6.1.4.1.14519.5.2.1.6279.6001.132817748896065918417924920957-660']

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 2  # wms 10->2
# if args.set=='cifar100':
#     CIFAR_CLASSES = 100

preprocesspath = args.crop_dir

pixvlu, npix = 0, 0
for fname in os.listdir(preprocesspath):
    # print(fname)
    if fname.endswith('.npy'):
        if fname[:-4] in blklst: continue
        data = np.load(os.path.join(preprocesspath, fname))
        pixvlu += np.sum(data)
        # print("data.shape = " + str(data.shape))
        npix += np.prod(data.shape)
pixmean = pixvlu / float(npix)
pixvlu = 0
for fname in os.listdir(preprocesspath):
    if fname.endswith('.npy'):
        if fname[:-4] in blklst: continue
        data = np.load(os.path.join(preprocesspath, fname)) - pixmean
        pixvlu += np.sum(data * data)
pixstd = np.sqrt(pixvlu / float(npix))
# pixstd /= 255
print(pixmean, pixstd)
logging.info('mean ' + str(pixmean) + ' std ' + str(pixstd))
# Datatransforms
logging.info('==> Preparing data..')  # Random Crop, Zero out, x z flip, scale,
transform_train = transforms.Compose([
    # transforms.RandomScale(range(28, 38)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomYFlip(),
    transforms.RandomZFlip(),
    transforms.ZeroOut(4),
    transforms.ToTensor(),
    transforms.Normalize((pixmean), (pixstd)),  # need to cal mean and std, revise norm func
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((pixmean), (pixstd)),
])

# load data list
trfnamelst = []
trlabellst = []
trfeatlst = []
tefnamelst = []
telabellst = []
tefeatlst = []

dataframe = pd.read_csv(args.annotations_path,
                        names=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])

alllst = dataframe['seriesuid'].tolist()[1:]
labellst = dataframe['malignant'].tolist()[1:]
crdxlst = dataframe['coordX'].tolist()[1:]
crdylst = dataframe['coordY'].tolist()[1:]
crdzlst = dataframe['coordZ'].tolist()[1:]
dimlst = dataframe['diameter_mm'].tolist()[1:]
# Published search split: subset0 updates architecture parameters and the
# other nine official subsets update supernet weights.
subset_ids = {}
for subset_index in range(10):
    subset_dir = os.path.join(args.luna_root, 'subset' + str(subset_index))
    if not os.path.isdir(subset_dir):
        raise FileNotFoundError('training subset directory does not exist: {}'.format(subset_dir))
    subset_ids[subset_index] = {
        fname[:-4] for fname in os.listdir(subset_dir) if fname.endswith('.mhd')
    }

architecture_ids = subset_ids[arch_subset]
weight_ids = set().union(*(ids for index, ids in subset_ids.items() if index != arch_subset))
overlap = architecture_ids & weight_ids
if overlap:
    raise RuntimeError('LUNA16 subset directories overlap: {}'.format(sorted(overlap)[:5]))

mxx = mxy = mxz = mxd = 0
for srsid, label, x, y, z, d in zip(alllst, labellst, crdxlst, crdylst, crdzlst, dimlst):
    mxx = max(abs(float(x)), mxx)
    mxy = max(abs(float(y)), mxy)
    mxz = max(abs(float(z)), mxz)
    mxd = max(abs(float(d)), mxd)
    if srsid in blklst: continue
    # crop raw pixel as feature
    data = np.load(os.path.join(preprocesspath, srsid + '.npy'))
    bgx = int(data.shape[0] / 2 - CROPSIZE / 2)
    bgy = int(data.shape[1] / 2 - CROPSIZE / 2)
    bgz = int(data.shape[2] / 2 - CROPSIZE / 2)
    data = np.array(data[bgx:bgx + CROPSIZE, bgy:bgy + CROPSIZE, bgz:bgz + CROPSIZE])
    # feat = np.hstack((np.reshape(data, (-1,)) / 255, float(d)))
    y, x, z = np.ogrid[-CROPSIZE / 2:CROPSIZE / 2, -CROPSIZE / 2:CROPSIZE / 2, -CROPSIZE / 2:CROPSIZE / 2]
    mask = abs(y ** 3 + x ** 3 + z ** 3) <= abs(float(d)) ** 3
    feat = np.zeros((CROPSIZE, CROPSIZE, CROPSIZE), dtype=float)
    feat[mask] = 1
    # print(feat.shape)
    series_id = srsid.split('-')[0]
    if series_id in architecture_ids:
        tefnamelst.append(srsid + '.npy')
        telabellst.append(int(label))
        tefeatlst.append(feat)
    elif series_id in weight_ids:
        trfnamelst.append(srsid + '.npy')
        trlabellst.append(int(label))
        trfeatlst.append(feat)
    else:
        raise RuntimeError('annotation series is absent from subset0-subset9: {}'.format(series_id))
for idx in range(len(trfeatlst)):
    # trfeatlst[idx][0] /= mxx
    # trfeatlst[idx][1] /= mxy
    # trfeatlst[idx][2] /= mxz
    trfeatlst[idx][-1] /= mxd
for idx in range(len(tefeatlst)):
    # tefeatlst[idx][0] /= mxx
    # tefeatlst[idx][1] /= mxy
    # tefeatlst[idx][2] /= mxz
    tefeatlst[idx][-1] /= mxd


def _architecture_tensors(model):
    return {
        name: getattr(model, name).detach().cpu()
        for name in ('alphas_normal', 'alphas_reduce', 'betas_normal', 'betas_reduce')
    }


def _genotype_dict(genotype):
    return {
        'normal': [list(item) for item in genotype.normal],
        'normal_concat': list(genotype.normal_concat),
        'reduce': [list(item) for item in genotype.reduce],
        'reduce_concat': list(genotype.reduce_concat),
    }


def save_search_state(path, model, optimizer, architect, lr_scheduler, loss_scaler, epoch, point):
    genotype = model.genotype()
    state = {
        'format_version': 1,
        'epoch': epoch,
        'point': point,
        'run_config': vars(args),
        'model_state_dict': model.state_dict(),
        'architecture': _architecture_tensors(model),
        'mask_normal': model.mask_normal.detach().cpu(),
        'mask_reduce': model.mask_reduce.detach().cpu(),
        'network_optimizer_state_dict': optimizer.state_dict(),
        'architecture_optimizer_state_dict': architect.optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'loss_scaler_state_dict': loss_scaler.state_dict(),
        'rng_state': capture_rng_state(),
        'genotype': _genotype_dict(genotype),
    }
    torch.save(state, path)
    with open(os.path.join(os.path.dirname(path), 'genotype.json'), 'w', encoding='utf-8') as stream:
        json.dump({'epoch': epoch, 'genotype': state['genotype']}, stream, indent=2)


def load_search_state(path, model, optimizer, architect, lr_scheduler, loss_scaler):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    required = {'model_state_dict', 'architecture', 'mask_normal', 'mask_reduce', 'epoch'}
    missing = required - set(checkpoint)
    if missing:
        raise KeyError('search checkpoint is missing fields: {}'.format(sorted(missing)))
    saved_config = checkpoint.get('run_config', {})
    for key in ('arch_subset', 'init_channels', 'layers', 'unrolled', 'pret', 'seed'):
        if key in saved_config and saved_config[key] != getattr(args, key):
            raise ValueError(
                'resume configuration mismatch for {}: checkpoint={!r}, command={!r}'.format(
                    key, saved_config[key], getattr(args, key)
                )
            )
    model.set_architecture_state(
        checkpoint['architecture'], checkpoint['mask_normal'], checkpoint['mask_reduce']
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if 'network_optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['network_optimizer_state_dict'])
    if 'architecture_optimizer_state_dict' in checkpoint:
        architect.optimizer.load_state_dict(checkpoint['architecture_optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if 'loss_scaler_state_dict' in checkpoint:
        loss_scaler.load_state_dict(checkpoint['loss_scaler_state_dict'])
    restore_rng_state(checkpoint.get('rng_state'))
    return int(checkpoint['epoch']) + 1, float(checkpoint.get('point', 0.2))


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    seed_everything(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.enabled = True
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    criterion_train = ConvSeparateLoss(weight=args.aux_loss_weight) if args.sep_loss == 'l2' else TriSeparateLoss(
        weight=args.aux_loss_weight)
    criterion_train = criterion_train.cuda()
    # criterion_val = nn.CrossEntropyLoss()
    # criterion_val = criterion_val.cuda()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, criterion_train)
    # print(model)
    torch.cuda.empty_cache()
    model = model.cuda()


    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     args.learning_rate,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay)
    optimizer = build_optimizer(config, model)
    loss_scaler = NativeScalerWithGradNormCount()

    trainset = lunanod(preprocesspath, trfnamelst, trlabellst, trfeatlst, train=True, download=True,
                       transform=transform_train)
    train_queue = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=0)  # change,20to0

    testset = lunanod(preprocesspath, tefnamelst, telabellst, tefeatlst, train=False, download=True,
                      transform=transform_test)
    valid_queue = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=0)  # change,20to0
    logging.info(
        'published search split: architecture=subset%d (%d nodules), weights=remaining subsets (%d nodules)',
        arch_subset, len(testset), len(trainset)
    )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(train_queue) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(train_queue))

    architect = Architect(model, args)
    point = 0.2
    start_epoch = 0
    if args.resume:
        start_epoch, point = load_search_state(
            args.resume, model, optimizer, architect, lr_scheduler, loss_scaler
        )
        logging.info('resumed search checkpoint %s at epoch %d', args.resume, start_epoch)
    for epoch in range(start_epoch, args.epochs):
        # lr = scheduler.get_last_lr()[0]
        lr = optimizer.param_groups[0]['lr']
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        # if epoch >= 0:
        if epoch >= pret:
            # print(F.softmax(model.alphas_normal, dim=-1))
            # print(F.softmax(model.alphas_reduce, dim=-1))
            # print(F.sigmoid(model.alphas_normal))
            # print(F.sigmoid(model.alphas_reduce))
            # logging.info("alphas_normal = \n{}".format(torch.sigmoid(model.alphas_normal)))
            # logging.info("alphas_reduce = \n{}".format(torch.sigmoid(model.alphas_reduce)))
            logging.info("alphas_normal = \n{}".format(torch.sigmoid(model.alphas_normal) * model.mask_normal))
            logging.info("alphas_reduce = \n{}".format(torch.sigmoid(model.alphas_reduce) * model.mask_reduce))
            logging.info("betas_normal = \n{}".format(torch.sigmoid(model.betas_normal)))
            logging.info("betas_reduce = \n{}".format(torch.sigmoid(model.betas_reduce)))
            # model.update_op()
        if epoch >= pret and epoch % 5 == 0:
            if args.epochs - epoch <= 10:
                point = 0.3
            elif args.epochs - epoch <= 20:
                point = 0.25
            model.update_op(point)
            # print(model)

        # test
        # point = 0.5
        # if epoch >= pret: #and epoch % 5 == 0:
        #     # if args.epochs - epoch <= 10:
        #     #     point = 0.5
        #     # elif args.epochs - epoch <= 20:
        #     #     point = 0.4
        #     model.update_op(point)

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, loss_scaler, lr_scheduler)
        logging.info('train_acc %f', train_acc)

        # validation
        # if args.epochs - epoch <= 200:        # wms+
        # if args.epochs - epoch <= -1000:
        if args.epochs - epoch <= 60:
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)

        # valid_acc, valid_obj = infer(valid_queue, model, criterion)
        # logging.info('valid_acc %f', valid_acc)

        save_search_state(
            os.path.join(args.save, 'search_checkpoint.pth.tar'),
            model, optimizer, architect, lr_scheduler, loss_scaler, epoch, point
        )
        # scheduler.step()


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, loss_scaler, lr_scheduler):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.train()
    optimizer.zero_grad()
    num_steps = len(train_queue)
    valid_queue_iter = cycling_batches(valid_queue)
    # top5 = utils.AvgrageMeter()

    for step, (input, target, feat) in enumerate(train_queue):
        # model.train()
        # torch.cuda.reset_peak_memory_stats()
        n = input.size(0)
        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(non_blocking=True)


        # Cycle through all subset0 architecture batches. Recreating the
        # iterator here would repeatedly select only its first batch.
        input_search, target_search, useless = next(valid_queue_iter)
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

        # if epoch >= 0:
        # if epoch >= 1000:
        if epoch >= pret:                  # wsm 原来是15
            # architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
            # objs_search, top1_search = architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
            if args.unrolled:
                # print("421312")
                # pass
                architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
            else:
                loss, loss1, loss2 = architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
            # logging.info('loss loss1 loss2 %f %f %f', loss, loss1, loss2)
            # if step % args.report_freq == 0 and step > 0:
            # logging.info('loss loss1 loss2 %f %f %f', loss, loss1, loss2)
            #     logging.info('search %03d %e %f', step, objs_search.avg, top1_search.avg)


        # optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            logits = model(input)
        loss = criterion(logits, target)

        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        # gradient = torch.autograd.grad(
        #     outputs=loss,
        #     inputs=list(model.parameters()),
        #     allow_unused=True,  # 设置 allow_unused=True
        # )
        #
        # # 获取模型的参数名称与参数值的映射
        # model_state_dict = model.state_dict()
        #
        # # 打印梯度为None的张量对应的参数名称
        # for i, grad in enumerate(gradient):
        #     if grad is None:
        #         param_name = list(model_state_dict.keys())[i]
        #         print(f"Gradient for parameter '{param_name}' is None")
        # # print(self.model.parameters())
        # print("123")

        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(step + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)

        if (step + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + step) // config.TRAIN.ACCUMULATION_STEPS)

        # torch.cuda.synchronize()



        # loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        # top5.update(prec5.item(), n)
        # if step > 20:
        #     return top1.avg, objs.avg
        if step % args.report_freq == 0 and step > 0:
            logging.info('train %03d %e %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    with torch.no_grad():
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        # top5 = utils.AvgrageMeter()
        model.eval()

        for step, (input, target, feat) in enumerate(valid_queue):
            # input = input.cuda()
            # target = target.cuda(non_blocking=True)
            # input = Variable(input, volatile=True).cuda()
            # target = Variable(target, volatile=True).cuda(non_blocking=True)

            input = Variable(input).cuda()
            target = Variable(target).cuda()
            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            # top5.update(prec5.item(), n)

            if step % args.report_freq == 0 and step > 0:
                logging.info('valid %03d %e %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
