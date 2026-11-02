import torch
import numpy as np
import os
import transforms as transforms
import pandas as pd
from dataloader import lunanod
from torch.autograd import Variable
from itertools import combinations, permutations
import logging
import pandas as pd
import argparse
import sys

import torch.nn.functional as F

import torch
import numpy as np
import os
from torch.autograd import Variable
from itertools import permutations
import pandas as pd
import argparse
from checkpointing import load_fassm_checkpoint


def load_data(luna_root, preprocess_path, annotations_path, fold, batch_size, num_workers):
    crop_size = 32
    black_list = []
    pix_value, npix = 0, 0
    for file_name in os.listdir(preprocess_path):
        if file_name.endswith('.npy'):
            if file_name[:-4] in black_list:
                continue
            data = np.load(os.path.join(preprocess_path, file_name))
            pix_value += np.sum(data)
            npix += np.prod(data.shape)
    pix_mean = pix_value / float(npix)
    pix_value = 0
    for file_name in os.listdir(preprocess_path):
        if file_name.endswith('.npy'):
            if file_name[:-4] in black_list: continue
            data = np.load(os.path.join(preprocess_path, file_name)) - pix_mean
            pix_value += np.sum(data * data)
    pix_std = np.sqrt(pix_value / float(npix))
    print(pix_mean, pix_std)
    transform_train = transforms.Compose([
        # transforms.RandomScale(range(28, 38)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomYFlip(),
        transforms.RandomZFlip(),
        transforms.ZeroOut(4),
        transforms.ToTensor(),
        transforms.Normalize((pix_mean), (pix_std)),  # need to cal mean and std, revise norm func
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((pix_mean), (pix_std)),
    ])

    # load data list
    test_file_name_list = []
    test_label_list = []
    test_feat_list = []

    data_frame = pd.read_csv(annotations_path,
                             names=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])

    all_list = data_frame['seriesuid'].tolist()[1:]
    label_list = data_frame['malignant'].tolist()[1:]
    crdx_list = data_frame['coordX'].tolist()[1:]
    crdy_list = data_frame['coordY'].tolist()[1:]
    crdz_list = data_frame['coordZ'].tolist()[1:]
    dim_list = data_frame['diameter_mm'].tolist()[1:]
    # test id
    test_id_list = []
    subset_dir = os.path.join(luna_root, 'subset' + str(fold))
    if not os.path.isdir(subset_dir):
        raise FileNotFoundError('test subset directory does not exist: {}'.format(subset_dir))
    for file_name in os.listdir(subset_dir):

        if file_name.endswith('.mhd'):
            test_id_list.append(file_name[:-4])
    mxx = mxy = mxz = mxd = 0
    for srsid, label, x, y, z, d in zip(all_list, label_list, crdx_list, crdy_list, crdz_list, dim_list):
        mxx = max(abs(float(x)), mxx)
        mxy = max(abs(float(y)), mxy)
        mxz = max(abs(float(z)), mxz)
        mxd = max(abs(float(d)), mxd)
        if srsid in black_list:
            continue
        # crop raw pixel as feature
        data = np.load(os.path.join(preprocess_path, srsid + '.npy'))
        bgx = int(data.shape[0] / 2 - crop_size / 2)
        bgy = int(data.shape[1] / 2 - crop_size / 2)
        bgz = int(data.shape[2] / 2 - crop_size / 2)
        data = np.array(data[bgx:bgx + crop_size, bgy:bgy + crop_size, bgz:bgz + crop_size])
        y, x, z = np.ogrid[-crop_size / 2:crop_size / 2, -crop_size / 2:crop_size / 2, -crop_size / 2:crop_size / 2]
        mask = abs(y ** 3 + x ** 3 + z ** 3) <= abs(float(d)) ** 3
        feat = np.zeros((crop_size, crop_size, crop_size), dtype=float)
        feat[mask] = 1
        if srsid.split('-')[0] in test_id_list:
            test_file_name_list.append(srsid + '.npy')
            test_label_list.append(int(label))
            test_feat_list.append(feat)
    for idx in range(len(test_feat_list)):
        test_feat_list[idx][-1] /= mxd
    print('preprocess_path:')
    print(preprocess_path)
    print('test_file_name_list:')
    print(test_file_name_list)
    test_set = lunanod(preprocess_path, test_file_name_list, test_label_list, test_feat_list, train=False,
                       download=True,
                       transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader


def load_module(path, allow_legacy_pickle=False):
    net, _ = load_fassm_checkpoint(
        path, torch.device('cuda'),
        defaults={'init_channels': 20, 'layers': 20, 'arch': 'DARTS_MAMBA',
                  'auxiliary': True, 'num_classes': 2},
        allow_legacy_pickle=allow_legacy_pickle,
    )
    return net


def get_targets(test_loader):
    target_list = np.empty(shape=0)
    for batch_idx, (inputs, targets, feat) in enumerate(test_loader):
        target_list = np.append(target_list, targets)
    target_list = target_list.astype(int)
    return target_list


def get_permutations(model_list, count, top_count):
    result = []
    for i in permutations(model_list, count):
        result.append(list(i))
        if result.__len__() >= top_count:
            return result
    return result


def T_module(module, test_loader):
    module.eval()
    total = 0
    correct = 0
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for batch_idx, (inputs, targets, feat) in enumerate(test_loader):
        print('batch_idx:')
        print(batch_idx)
        inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs, requires_grad=False), Variable(targets)
        outputs, logits_aux, _ = module(inputs)
        prediction = 0

        if not isinstance(outputs, tuple):

            confidence, prediction = torch.max(outputs.data.softmax(dim=1), 1)# 將output的data數據，用softmax轉換為概率 dim = 1是每行處理

            print('confidence:')
            print(confidence)

            confidence_two = outputs.data.softmax(dim=1)
            print('confidence_two:')
            print(confidence_two)

            print('outputs.data:')
            print(outputs.data)

            print('prediction:')
            print(prediction)

            print('targets.data:')
            print(targets.data)


        else:
            confidence, prediction = torch.max(outputs[0].data, 1)

        TP += ((prediction == 1) & (targets.data == 1)).cpu().sum()
        TN += ((prediction == 0) & (targets.data == 0)).cpu().sum()
        FN += ((prediction == 0) & (targets.data == 1)).cpu().sum()
        FP += ((prediction == 1) & (targets.data == 0)).cpu().sum()
    tpr = 100. * TP.data.item() / (TP.data.item() + FN.data.item())
    fpr = 100. * FP.data.item() / (FP.data.item() + TN.data.item())
    acc = (TP.data.item()+TN.data.item()) / (TP.data.item()+TN.data.item()+FN.data.item()+FP.data.item())
    tnr = 100. * TN.data.item() / (TN.data.item() + FP.data.item())
    fnr = 100. * FN.data.item() / (TP.data.item() + FN.data.item())
    F1score = 100. * (TP.data.item() + TP.data.item()) / (
        TP.data.item() + TP.data.item() + FN.data.item() + FP.data.item())

    Gmean = np.sqrt(tpr * tnr)

    print(f'acc:{acc}')
    print('tpr ' + str(tpr) + ' fpr ' + str(fpr))
    print('tnr ' + str(tnr) + ' fnr ' + str(fnr))

    print('F1score ' + str(F1score))

    print('Gmean' + str(Gmean))



def get_predicted(result_array):
    positive_array = result_array == 1
    negative_array = result_array == 0
    positive_count = np.sum(positive_array, axis=0)
    negative_count = np.sum(negative_array, axis=0)
    predicted = positive_count > negative_count
    return predicted.astype(int)


parser = argparse.ArgumentParser(description='test')
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
parser.add_argument('--model_path', type=str, required=True, help='checkpoint to evaluate')
parser.add_argument('--fold', type=int, required=True, choices=range(5, 10),
                    help='held-out paper fold')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--luna_root', type=str, required=True,
                    help='directory containing LUNA16 subset0 through subset9 directories')
parser.add_argument('--crop_dir', type=str, required=True,
                    help='directory containing the preprocessed 32^3 nodule .npy crops')
parser.add_argument('--annotations_path', type=str,
                    default=os.path.join(_REPO_ROOT, 'data', 'annotationdetclsconvfnl_v3.csv'),
                    help='CSV containing seriesuid, coordinates, diameter, and malignancy labels')
parser.add_argument('--allow_legacy_pickle', action='store_true',
                    help='allow trusted legacy checkpoints that pickle the full model')

args = parser.parse_args()

if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA-capable GPU is required for confidence evaluation')
    for option, path in (('--model_path', args.model_path), ('--luna_root', args.luna_root),
                         ('--crop_dir', args.crop_dir), ('--annotations_path', args.annotations_path)):
        if not os.path.exists(path):
            parser.error('{} must be an existing path: {}'.format(option, path))
    fold = args.fold
    batch_size = args.batch_size
    num_workers = args.num_workers
    luna_root = args.luna_root
    preprocess_path = args.crop_dir
    net = load_module(args.model_path, args.allow_legacy_pickle)
    test_data_loader = load_data(luna_root, preprocess_path, args.annotations_path, fold, batch_size, num_workers)
    T_module(net, test_data_loader)
