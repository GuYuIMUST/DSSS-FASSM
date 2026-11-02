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
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkLUNA as Network

import ssl

# from __future__ import print_function

# import torch
# import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn

import transforms as transforms
from dataloader import lunanod, lunanod_2
import os
# import argparse
# import time
# from utils import progress_bar
# from torch.autograd import Variable
import logging
# import numpy as np
import ast
import pandas as pd
import random

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

from config import get_config
# from models import build_model
# from data import build_loader
from utils_mamba.lr_scheduler import build_scheduler
from utils_mamba.optimizer import build_optimizer
from utils_mamba.logger import create_logger
from utils_mamba.utils import NativeScalerWithGradNormCount, auto_resume_helper, reduce_tensor
from utils_mamba.utils import load_checkpoint_ema, load_pretrained_ema, save_checkpoint_ema
from repro_utils import capture_rng_state, restore_rng_state, seed_everything

from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count

from timm.utils import ModelEma as ModelEma

# if torch.multiprocessing.get_start_method() != "spawn":
#     print(f"||{torch.multiprocessing.get_start_method()}||", end="")
#     torch.multiprocessing.set_start_method("spawn", force=True)

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
# 收敛重要设置
# parser.add_argument('--batch_size', type=int, default=12, help='batch size')
# parser.add_argument('--batch_size', type=int, default=4, help='batch size')
# new
# parser.add_argument('--batch_size', type=int, default=10, help='batch size')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
# parser.add_argument('--batch_size', type=int, default=64, help='batch size')
# parser.add_argument('--batch_size', type=int, default=8, help='batch size')

# parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate', type=float, default=2e-4,
                    help='published base learning rate (mirrored by config.py)')

# parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')

parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
# parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
# parser.add_argument('--init_channels', type=int, default=96, help='num of init channels')
# parser.add_argument('--init_channels', type=int, default=23, help='num of init channels')
parser.add_argument('--init_channels', type=int, default=20, help='num of init channels')
# parser.add_argument('--init_channels', type=int, default=32, help='num of init channels')

# parser.add_argument('--layers', type=int, default=20, help='total number of layers')  # 20
parser.add_argument('--layers', type=int, default=20, help='total number of layers')

parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
# parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary', action=argparse.BooleanOptionalAction, default=True,
                    help='use the two training-only auxiliary classifiers')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS_MAMBA', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

# parser.add_argument('--fold', type=int, default=5, help="fold")

# origin Lu
# parser.add_argument('--fold_validate', type=int, default=3, help="fold for validate")
# parser.add_argument('--fold_test', type=int, default=5, help="fold for test")

# To control variables
parser.add_argument('--fold_validate', type=int, default=9, choices=range(5, 10),
                    help="held-out paper fold; train on the other nine subsets")
# parser.add_argument('--fold_test', type=int, default=3, help="fold for test")

parser.add_argument('--test_flag', action='store_true', help="resume from --pretrain_model_path")
parser.add_argument('--pretrain_model_path', type=str, default='', help='checkpoint used with --test_flag')
parser.add_argument('--allow_legacy_pickle', action='store_true',
                    help='allow a trusted legacy checkpoint that pickles a complete model')

args = parser.parse_args()

for option, path in (('--crop_dir', args.crop_dir), ('--luna_root', args.luna_root)):
    if not os.path.isdir(path):
        parser.error('{} must be an existing directory: {}'.format(option, path))
if not os.path.isfile(args.annotations_path):
    parser.error('--annotations_path must be an existing file: {}'.format(args.annotations_path))
if args.test_flag and not args.pretrain_model_path:
    parser.error('--test_flag requires --pretrain_model_path')
if args.test_flag and not os.path.isfile(args.pretrain_model_path):
    parser.error('--pretrain_model_path must be an existing file: {}'.format(args.pretrain_model_path))

config = get_config(args)

best_acc = 0
CROPSIZE = 32
gbtdepth = 1
fold_validate = args.fold_validate
# fold_test = args.fold_test
# fold = args.fold
blklst = []  # ['1.3.6.1.4.1.14519.5.2.1.6279.6001.121993590721161347818774929286-388', \
# '1.3.6.1.4.1.14519.5.2.1.6279.6001.121993590721161347818774929286-389', \
# '1.3.6.1.4.1.14519.5.2.1.6279.6001.132817748896065918417924920957-660']


args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# CIFAR_CLASSES = 10
CIFAR_CLASSES = 2

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

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((pixmean), (pixstd)),
# ])

transform_validate = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((pixmean), (pixstd)),
])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((pixmean), (pixstd)),
# ])

# load data list
trfnamelst = []
trlabellst = []
trfeatlst = []
# tefnamelst = []
# telabellst = []
# tefeatlst = []
valfnamelst = []
vallabellst = []
valfeatlst = []

# tesfnamelst = []
# teslabellst = []
# tesfeatlst = []

dataframe = pd.read_csv(args.annotations_path,
                        names=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])

alllst = dataframe['seriesuid'].tolist()[1:]
labellst = dataframe['malignant'].tolist()[1:]
crdxlst = dataframe['coordX'].tolist()[1:]
crdylst = dataframe['coordY'].tolist()[1:]
crdzlst = dataframe['coordZ'].tolist()[1:]
dimlst = dataframe['diameter_mm'].tolist()[1:]

# validate id
validate_idlst = [] #liu validate
validate_subset_dir = os.path.join(args.luna_root, 'subset' + str(fold_validate))
if not os.path.isdir(validate_subset_dir):
    raise FileNotFoundError('validation subset directory does not exist: {}'.format(validate_subset_dir))
for fname in os.listdir(validate_subset_dir):

    if fname.endswith('.mhd'):#liu subset以.mhd結尾
        validate_idlst.append(fname[:-4]) #liu 除了倒数第四个数取全部

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
#     if srsid.split('-')[0] in teidlst:
#         tefnamelst.append(srsid + '.npy')
#         telabellst.append(int(label))
#         tefeatlst.append(feat)
#     else:
#         trfnamelst.append(srsid + '.npy')
#         trlabellst.append(int(label))
#         trfeatlst.append(feat)
# for idx in range(len(trfeatlst)):
#     # trfeatlst[idx][0] /= mxx
#     # trfeatlst[idx][1] /= mxy
#     # trfeatlst[idx][2] /= mxz
#     trfeatlst[idx][-1] /= mxd
# for idx in range(len(tefeatlst)):
#     # tefeatlst[idx][0] /= mxx
#     # tefeatlst[idx][1] /= mxy
#     # tefeatlst[idx][2] /= mxz
#     tefeatlst[idx][-1] /= mxd
    if srsid.split('-')[0] in validate_idlst:
        valfnamelst.append(srsid + '.npy')
        vallabellst.append(int(label))
        valfeatlst.append(feat)
    # elif srsid.split('-')[0] in test_idlst:
    #     tesfnamelst.append(srsid + '.npy')
    #     teslabellst.append(int(label))
    #     tesfeatlst.append(feat)
    else:
        trfnamelst.append(srsid + '.npy')
        trlabellst.append(int(label))
        trfeatlst.append(feat)
for idx in range(len(trfeatlst)):
    # trfeatlst[idx][0] /= mxx
    # trfeatlst[idx][1] /= mxy
    # trfeatlst[idx][2] /= mxz
    trfeatlst[idx][-1] /= mxd
for idx in range(len(valfeatlst)):
    # tefeatlst[idx][0] /= mxx
    # tefeatlst[idx][1] /= mxy
    # tefeatlst[idx][2] /= mxz
    valfeatlst[idx][-1] /= mxd
# for idx in range(len(tesfeatlst)):
#     # tefeatlst[idx][0] /= mxx
#     # tefeatlst[idx][1] /= mxy
#     # tefeatlst[idx][2] /= mxz
#     valfeatlst[idx][-1] /= mxd

def main():
    global best_acc
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    seed_everything(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.enabled = True
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    logging.info("genotype = %s", genotype)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))



    # # imbalanced learning
    # # weight = torch.tensor([1.0, 5.0], dtype=torch.float32)  # 这里的 9.0 是阴性类的权重，1.0 是阳性类的权重
    # weight = torch.tensor([0.2, 1.0], dtype=torch.float32)  # 这里的 9.0 是阴性类的权重，1.0 是阳性类的权重
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weight = weight.to(device)
    #
    # # 创建 CrossEntropyLoss 对象，并传入 weight
    # criterion = torch.nn.CrossEntropyLoss(weight=weight)



    criterion = nn.CrossEntropyLoss()

    criterion = criterion.cuda()
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     args.learning_rate,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay
    # )
    optimizer = build_optimizer(config, model)
    loss_scaler = NativeScalerWithGradNormCount()

    # train_transform, valid_transform = utils._data_transforms_cifar10(args)
    # train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    # valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    #
    # train_queue = torch.utils.data.DataLoader(
    #     train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    #
    # valid_queue = torch.utils.data.DataLoader(
    #     valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    trainset = lunanod(preprocesspath, trfnamelst, trlabellst, trfeatlst, train=True, download=True, transform=transform_train)
    # trainset = lunanod_2(preprocesspath, trfnamelst, trlabellst, trfeatlst, train=True, download=True, transform=transform_train)
    # print(len(trainset))
    # train_queue = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)  # change,20to0
    train_queue = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # testset = lunanod(preprocesspath, tefnamelst, telabellst, tefeatlst, train=False, download=True,
    #                   transform=transform_test)
    # valid_queue = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
    #                                           num_workers=0)  # change,20to0

    validateset = lunanod(preprocesspath, valfnamelst, vallabellst, valfeatlst, train=False, download=True, transform=transform_validate)
    valid_queue = torch.utils.data.DataLoader(validateset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # testdateset = lunanod(preprocesspath, tesfnamelst, teslabellst, tesfeatlst, train=False, download=True, transform=transform_test)
    # test_queue = torch.utils.data.DataLoader(testdateset, batch_size=args.batch_size, shuffle=False, num_workers=0)



    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(train_queue) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(train_queue))

    # The model can continue running the next day
    test_flag = args.test_flag
    pretrain_model_path = args.pretrain_model_path
    start_epoch = 0
    if test_flag:
        try:
            checkpoint = torch.load(pretrain_model_path, map_location='cpu', weights_only=True)
        except Exception:
            if not args.allow_legacy_pickle:
                raise ValueError(
                    'checkpoint requires unsafe legacy pickle loading; pass '
                    '--allow_legacy_pickle only when the checkpoint is trusted'
                )
            checkpoint = torch.load(pretrain_model_path, map_location='cpu', weights_only=False)
        saved_config = checkpoint.get('run_config', {}) if isinstance(checkpoint, dict) else {}
        for key in ('fold_validate', 'arch', 'init_channels', 'layers', 'auxiliary'):
            if key in saved_config and saved_config[key] != getattr(args, key):
                raise ValueError(
                    'resume configuration mismatch for {}: checkpoint={!r}, command={!r}'.format(
                        key, saved_config[key], getattr(args, key)
                    )
                )
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint)) \
            if isinstance(checkpoint, dict) else checkpoint
        if isinstance(state_dict, nn.Module):
            state_dict = state_dict.state_dict()
        model.load_state_dict(state_dict)
        if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if isinstance(checkpoint, dict) and 'scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if isinstance(checkpoint, dict) and 'loss_scaler_state_dict' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['loss_scaler_state_dict'])
        if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            start_epoch = int(checkpoint['epoch']) + 1
        if isinstance(checkpoint, dict) and 'best_acc' in checkpoint:
            best_acc = float(checkpoint['best_acc'])
        if isinstance(checkpoint, dict):
            restore_rng_state(checkpoint.get('rng_state'))
        logging.info('loaded checkpoint %s; resuming at epoch %d', pretrain_model_path, start_epoch)
        #     logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
        #     model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        #
        #     train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        #     # logging.info('train_acc %f', train_acc)
        #
        #     valid_acc, valid_obj = infer(valid_queue, model, criterion)
        #     logging.info('valid_acc %f', valid_acc)
        #
        #     scheduler.step()
        #     utils.save_2(model, optimizer, scheduler, epoch, os.path.join(args.save, 'weights.pt'))
        #     if valid_acc > best_acc:
        #     # if valid_acc > 89:
        #         state = {
        #             'model': model,
        #             'valid_acc': valid_acc,
        #             'epoch': epoch,
        #         }
        #         torch.save(state, os.path.join(args.save, str(valid_acc) + '_' + str(epoch) + '_model.pt'))
        #         best_acc = valid_acc

    if start_epoch < args.epochs:
        for epoch in range(start_epoch, args.epochs):
            # scheduler.step()
            # logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
            logging.info('epoch %d lr %e', epoch, optimizer.param_groups[0]['lr'])
            model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

            train_acc, train_obj = train(train_queue, model, criterion, optimizer, loss_scaler, lr_scheduler, epoch)
            logging.info('train_acc %f', train_acc)

            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)

            # test_acc, test_obj = infer(test_queue, model, criterion)
            # logging.info('test_acc %f', test_acc)

            # scheduler.step()
            # utils.save_2(model, optimizer, scheduler, epoch, os.path.join(args.save, 'weights.pt'))

            is_best = valid_acc > best_acc
            if is_best:
                best_acc = valid_acc
            state = {
                'format_version': 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'loss_scaler_state_dict': loss_scaler.state_dict(),
                'rng_state': capture_rng_state(),
                'valid_acc': valid_acc,
                'epoch': epoch,
                'best_acc': best_acc,
                'run_config': vars(args),
                'genotype': repr(genotype),
            }
            torch.save(state, os.path.join(args.save, 'last_model.pth.tar'))
            if is_best:
                torch.save(state, os.path.join(args.save, 'best_model.pth.tar'))


def train(train_queue, model, criterion, optimizer, loss_scaler, lr_scheduler, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    correct = 0
    total = 0
    optimizer.zero_grad()
    num_steps = len(train_queue)

    for step, (input, target, feat) in enumerate(train_queue):
        # input = Variable(input).cuda()
        # target = Variable(target).cuda(async=True)
        # target = Variable(target).cuda()
        torch.cuda.reset_peak_memory_stats()
        # input = Variable(input, requires_grad=False).cuda()
        # target = Variable(target, requires_grad=False).cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
        # with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            # logits, logits_aux = model(input)
            logits, logits_aux, logits_aux0 = model(input)
            loss = criterion(logits, target)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            if args.auxiliary:
                loss_aux = criterion(logits_aux, target)
                loss_aux0 = criterion(logits_aux0, target)
                loss += args.auxiliary_weight * loss_aux
                loss += args.auxiliary_weight * loss_aux0

        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(step + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)

        if (step + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + step) // config.TRAIN.ACCUMULATION_STEPS)

        torch.cuda.synchronize()
        # loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0 and step > 0:
            # logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            logging.info('train %03d %e %f', step, objs.avg, top1.avg)
        _, predicted = torch.max(logits.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
    logging.info('train %03d %e %f', step, objs.avg, top1.avg)
    logging.info('tracc_TEST ' + str(correct.data.item() / float(total) * 100.))
    # print('ep ' + str(epoch) + ' tracc ' + str(correct.data.item() / float(total)) + ' lr ' + str(lr))
    #
    # logging.info(
    #     'ep ' + str(epoch) + ' tracc ' + str(correct.data.item() / float(total)) + ' lr ' + str(lr))

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    with torch.no_grad():
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.eval()

        correct = 0
        total = 0
        TP = FP = FN = TN = 0

    # for step, (input, target) in enumerate(valid_queue):
    # with torch.no_grad():
        for step, (input, target, feat) in enumerate(valid_queue):
            # with torch.no_grad():
            input = Variable(input).cuda()
            target = Variable(target).cuda()
            # input = Variable(input, volatile=True).cuda()
            # target = Variable(target, volatile=True).cuda(async=True)
            # target = Variable(target, volatile=True).cuda()
            # input = Variable(input, volatile=True).cuda()
            # target = Variable(target, volatile=True).cuda(non_blocking=True)

            logits, _, _ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()

            TP += ((predicted == 1) & (target.data == 1)).cpu().sum()
            TN += ((predicted == 0) & (target.data == 0)).cpu().sum()
            FN += ((predicted == 0) & (target.data == 1)).cpu().sum()
            FP += ((predicted == 1) & (target.data == 0)).cpu().sum()

            # if step % args.report_freq == 0:
            #     logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        tpr = 100. * TP.data.item() / (TP.data.item() + FN.data.item())
        fpr = 100. * FP.data.item() / (FP.data.item() + TN.data.item())
        Accuracy = 100. * (TP.data.item() + TN.data.item())/ (TP.data.item() + FN.data.item() + TN.data.item() + FP.data.item())
        Sensitivity = 100. * TP.data.item() / (TP.data.item() + FN.data.item())
        Specificity = 100. * TN.data.item() / (FP.data.item() + TN.data.item())
        F1_Score = 100. * (2. * TP.data.item()) / (2. * TP.data.item() + FP.data.item() + FN.data.item())
        # Precision = TP.data.item() / (TP.data.item() + FP.data.item())
        # Recall = TP.data.item() / (TP.data.item() + FN.data.item())
        # F1_Score_2 = 100. * (2. * Precision * Recall) / (Precision + Recall)
        logging.info('tpr ' + str(tpr) + ' fpr ' + str(fpr))
        logging.info('Accuracy ' + str(Accuracy) + ' Sensitivity ' + str(Sensitivity) + ' Specificity ' + str(Specificity) + ' F1_Score ' + str(F1_Score))
        logging.info('valid %03d %e %f', step, objs.avg, top1.avg)
    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
