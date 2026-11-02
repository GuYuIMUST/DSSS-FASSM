import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from utils import drop_path


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_B, dropout_rate):
        super(Cell, self).__init__()
        self.dropout_rate = dropout_rate
        # if reduction_B:
        #     # self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        #     self.preprocess0 = PatchMerging3D(
        #         C_prev_prev,
        #         C,
        #         channel_first=True)
        # else:
        #     self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        # self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction_B:
            self.preprocess0 = PatchMerging3D(
                C_prev_prev,
                C,
                channel_first=True)
        elif reduction:
            self.preprocess0 = PatchMerging3D(
                C_prev_prev,
                C,
                channel_first=True)
        else:
            # 只做了chanel变化，map没变
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)

        if reduction:
            self.preprocess1 = PatchMerging3D(
                C_prev,
                C,
                channel_first=True)
        else:
            # 只做了chanel变化，map没变
            self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            # op = OPS[name](C, stride, True)
            op = OPS[name](C, stride, self.dropout_rate)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadLUNA(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadLUNA, self).__init__()
        # self.features = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     nn.AvgPool3d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
        #     nn.Conv3d(C, 128, 1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(128, 768, 2, bias=False),
        #     nn.BatchNorm3d(768),
        #     nn.ReLU(inplace=True)
        # )
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.AvgPool3d(3, stride=1, padding=1, count_include_pad=False),  # image size = 2 x 2
            nn.AdaptiveAvgPool3d(2),
            nn.Conv3d(C, 128, 1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 768, 2, bias=False),
            nn.BatchNorm3d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x

class AuxiliaryHeadLUNA0(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadLUNA0, self).__init__()
        # self.features = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     nn.AvgPool3d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
        #     nn.Conv3d(C, 128, 1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(128, 768, 2, bias=False),
        #     nn.BatchNorm3d(768),
        #     nn.ReLU(inplace=True)
        # )
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.AvgPool3d(3, stride=1, padding=1, count_include_pad=False),  # image size = 2 x 2
            nn.AdaptiveAvgPool3d(2),
            nn.Conv3d(C, 128, 1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 768, 2, bias=False),
            nn.BatchNorm3d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x
# class AuxiliaryHeadImageNet(nn.Module):
#
#     def __init__(self, C, num_classes):
#         """assuming input size 14x14"""
#         super(AuxiliaryHeadImageNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
#             nn.Conv2d(C, 128, 1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 768, 2, bias=False),
#             # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
#             # Commenting it out for consistency with the experiments in the paper.
#             # nn.BatchNorm2d(768),
#             nn.ReLU(inplace=True)
#         )
#         self.classifier = nn.Linear(768, num_classes)
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x.view(x.size(0), -1))
#         return x


class NetworkLUNA(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkLUNA, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        stem_multiplier = 1
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            # nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            # nn.BatchNorm2d(C_curr)
            nn.Conv3d(1, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm3d(C_curr)
        )
        # self.patch_embed = self._make_patch_embed(1, C_curr, 2, True, LayerNorm2d, channel_first=True)

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False

        # self.dpr = [x.item() for x in torch.linspace(0, 0.5, layers)]
        self.dpr = [x.item() for x in torch.linspace(0, 0, layers)]

        for i in range(layers):
            # self.dpr[i]
            # if i in [layers // 3, 2 * layers // 3]:
            if i in [1, 3]:
            # if i in [2, 5]:
            # if i in [1, 3, layers-2]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.dpr[i])
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

            if i == layers // 3:
                C_to_auxiliary0 = C_prev
        # print(self.cells)
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadLUNA(C_to_auxiliary, num_classes)

            self.auxiliary_head0 = AuxiliaryHeadLUNA0(C_to_auxiliary0, num_classes)
        # self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(C_prev, num_classes)


    @staticmethod
    def _make_patch_embed(in_chans=1, embed_dim=96, patch_size=2, patch_norm=True, norm_layer=LayerNorm2d,
                          channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        # return nn.Sequential(
        #     nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
        #     (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
        #     (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        # )
        return nn.Sequential(
            nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            # (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 4, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )


    def forward(self, input):
        logits_aux = None
        logits_aux0 = None
        # logits_aux1 = None
        s0 = s1 = self.stem(input)
        # s0 = s1 = self.patch_embed(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux0 = self.auxiliary_head0(s1)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        # if self._auxiliary and self.training:
        #     logits_aux = logits_aux0 + logits_aux1
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux, logits_aux0


# class NetworkImageNet(nn.Module):
#
#     def __init__(self, C, num_classes, layers, auxiliary, genotype):
#         super(NetworkImageNet, self).__init__()
#         self._layers = layers
#         self._auxiliary = auxiliary
#
#         self.stem0 = nn.Sequential(
#             nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(C // 2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(C),
#         )
#
#         self.stem1 = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(C),
#         )
#
#         C_prev_prev, C_prev, C_curr = C, C, C
#
#         self.cells = nn.ModuleList()
#         reduction_prev = True
#         for i in range(layers):
#             if i in [layers // 3, 2 * layers // 3]:
#                 C_curr *= 2
#                 reduction = True
#             else:
#                 reduction = False
#             cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
#             reduction_prev = reduction
#             self.cells += [cell]
#             C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
#             if i == 2 * layers // 3:
#                 C_to_auxiliary = C_prev
#
#         if auxiliary:
#             self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
#         self.global_pooling = nn.AvgPool2d(7)
#         self.classifier = nn.Linear(C_prev, num_classes)
#
#     def forward(self, input):
#         logits_aux = None
#         s0 = self.stem0(input)
#         s1 = self.stem1(s0)
#         for i, cell in enumerate(self.cells):
#             s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
#             if i == 2 * self._layers // 3:
#                 if self._auxiliary and self.training:
#                     logits_aux = self.auxiliary_head(s1)
#         out = self.global_pooling(s1)
#         logits = self.classifier(out.view(out.size(0), -1))
#         return logits, logits_aux
