import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
# train speed is slower after enabling this opts.
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

try:
    from models_mamba.csm_triton import cross_scan_fn, cross_merge_fn
except:
    from models_mamba.csm_triton import cross_scan_fn, cross_merge_fn

try:
    from models_mamba.csms6s import selective_scan_fn, selective_scan_flop_jit
except:
    from models_mamba.csms6s import selective_scan_fn, selective_scan_flop_jit

# FLOPs counter not prepared fro mamba2
try:
    from models_mamba.mamba2.ssd_minimal import selective_scan_chunk_fn
except:
    from models_mamba.mamba2.ssd_minimal import selective_scan_chunk_fn


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride, ::self.stride].mul(0.)


class MixedOp(nn.Module):

    def __init__(self, dim, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.stride = stride
        # self.p = Zero(stride)

        # self.mp = nn.MaxPool3d(2, 2)  # wms 2改3
        # self.k = 4
        # for primitive in PRIMITIVES:
        #     op = OPS[primitive](C // self.k, stride, False)
        #     if 'pool' in primitive:
        #         op = nn.Sequential(op, nn.BatchNorm3d(C // self.k, affine=False))  # wms 2改3
        #     self._ops.append(op)
        for primitive in PRIMITIVES:
            op = OPS[primitive](dim, stride, 0)
            self._ops.append(op)

    # def update_p(self):
    #     for op in self._ops:
    #         if isinstance(op, nn.Sequential):
    #             if isinstance(op[0], Identity):
    #                 op[1].p = self.p

    def replace_ops_with_zero(self, weights, point=0.3):
        # mask = weights >= 0.25
        # mask = weights >= 0.3
        mask = weights >= point
        new_weights = weights * mask.float()
        for i, weight in enumerate(new_weights):
            if weight.item() == 0.0 and not isinstance(self._ops[i], Zero):
                # Replace the operation with a zero operation
                # self._ops[i] = Zero(self.stride)
                self._ops[i] = Zero(stride=1)

    def apply_mask(self, mask):
        """Irreversibly remove operations disabled by a saved KO mask."""
        for i, keep in enumerate(mask):
            if not bool(keep.item()) and not isinstance(self._ops[i], Zero):
                self._ops[i] = Zero(stride=1)

    def forward(self, x, weights):
        # # channel proportion k=4
        # dim_2 = x.shape[1]
        # xtemp = x[:, :  dim_2 // self.k, :, :, :]  # wms +   , :
        # xtemp2 = x[:, dim_2 // self.k:, :, :, :]  # wms +   , :
        # temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))
        # # reduction cell needs pooling before concat
        # if temp1.shape[2] == x.shape[2]:
        #     ans = torch.cat([temp1, xtemp2], dim=1)
        # else:
        #     ans = torch.cat([temp1, self.mp(xtemp2)], dim=1)
        # ans = channel_shuffle(ans, self.k)
        # return ans
        # weights[weights < 0.5] = 0

        # mask = weights >= 0.25
        # new_weights = weights * mask.float()
        # return sum(w * op(x) for w, op in zip(new_weights, self._ops))

        return sum(w * op(x) for w, op in zip(weights, self._ops))

class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_B):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_B:
            # self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
            # downsample
            # downsample = PatchMerging2D(
            #     self.dims[i_layer],
            #     self.dims[i_layer + 1],
            #     norm_layer=norm_layer,
            #     channel_first=self.channel_first,
            # ) if (i_layer < self.num_layers - 1) else nn.Identity()
            # chanel和map都变了，在next reduction cell

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
        # self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
                for j in range(2 + i):
                    # 这个才是在reduction cell
                    stride = 2 if reduction and j < 2 else 1
                    op = MixedOp(C, stride)
                    self._ops.append(op)

    def update_op(self, weights, point):
        # for op in self._ops:

        for index, op in enumerate(self._ops):
            op.replace_ops_with_zero(weights[index], point)

    def apply_mask(self, mask):
        for index, op in enumerate(self._ops):
            op.apply_mask(mask[index])

    def forward(self, s0, s1, weights, weights2):
        s0 = self.preprocess0(s0).contiguous()
        s1 = self.preprocess1(s1).contiguous()

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(weights2[offset + j] * self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, criterion2, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._criterion2 = criterion2
        self._steps = steps
        self._multiplier = multiplier
        # self._exchange = Permute(0, 4, 1, 2, 3)
        # C_curr = stem_multiplier * C
        # C = 96
        C_curr = 96
        # C_curr = 32
        self.dims_embed = C_curr

        # self.stem = nn.Sequential(
        #     nn.Conv3d(1, C_curr, 3, padding=1, bias=False),
        #     nn.BatchNorm3d(C_curr)
        # )

        # C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C_curr
        self.cells = nn.ModuleList()
        reduction_B = False
        for i in range(layers):
            # if i in [layers // 3, 2 * layers // 3]:
            if i in [1, 3]:
            # if i in [1, 3, layers-2]:
            # if i in [10, 11]:
            # if i in [layers // 4, 2 * layers // 4, 3 * layers // 4]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_B)
            reduction_B = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        # self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()
        # rweights = torch.sigmoid(self.alphas_reduce)
        # nweights = torch.sigmoid(self.alphas_normal)
        # mask1 = rweights >= point
        # self.alphas_reduce = (rweights * mask1.float()).detach()
        # mask2 = nweights >= point
        # self.alphas_normal = (nweights * mask2.float()).detach()
        self.mask_normal = torch.ones_like(self.alphas_normal)
        self.mask_reduce = torch.ones_like(self.alphas_reduce)

        self.in_chans = 1
        # self.dims_embed = 96
        self.patch_size = 2
        self.patch_norm = True
        self.norm_layer = LayerNorm2d
        self.patch_embed = self._make_patch_embed(self.in_chans, self.dims_embed, self.patch_size, self.patch_norm,
                                                  self.norm_layer, channel_first=True)
        self.channel_first = True
        self.classifier = nn.Sequential(OrderedDict(
            norm=self.norm_layer(C_prev),  # B,H,W,C
            # permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            permute=(Permute(0, 4, 1, 2, 3) if not self.channel_first else nn.Identity()),
            # avgpool=nn.AdaptiveAvgPool2d(1),
            avgpool=nn.AdaptiveAvgPool3d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(C_prev, num_classes),
        ))
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    #
    # # used in building optimizer
    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {"pos_embed"}
    #
    # # used in building optimizer
    # @torch.jit.ignore
    # def no_weight_decay_keywords(self):
    #     return {}

    def new(self):
        # Preserve the irreversible KO structure in the one-step unrolled
        # model. A freshly constructed Network would restore pruned modules.
        return copy.deepcopy(self).cuda()

    def forward(self, input):
        # s0 = s1 = self.stem(input)
        s0 = s1 = self.patch_embed(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                # weights = torch.sigmoid(self.alphas_reduce)
                weights = torch.sigmoid(self.alphas_reduce) * self.mask_reduce
                n = 3
                start = 2
                weights2 = torch.sigmoid(self.betas_reduce[0:2])
                for i in range(self._steps - 1):
                    end = start + n
                    tw2 = torch.sigmoid(self.betas_reduce[start:end])
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2, tw2], dim=0)
            else:
                # weights = torch.sigmoid(self.alphas_normal)
                weights = torch.sigmoid(self.alphas_normal) * self.mask_normal
                n = 3
                start = 2
                weights2 = torch.sigmoid(self.betas_normal[0:2])
                for i in range(self._steps - 1):
                    end = start + n
                    tw2 = torch.sigmoid(self.betas_normal[start:end])
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2, tw2], dim=0)
            s0, s1 = s1, cell(s0, s1, weights, weights2)
        # out = self.global_pooling(s1)
        # logits = self.classifier(out.view(out.size(0), -1))
        logits = self.classifier(s1)
        return logits

    def update_op(self, point):
        rweights = torch.sigmoid(self.alphas_reduce)
        nweights = torch.sigmoid(self.alphas_normal)
        # mask1 = rweights >= point
        # self.alphas_reduce = (rweights * mask1.float()).detach()
        # mask2 = nweights >= point
        # self.alphas_normal = (nweights * mask2.float()).detach()

        # KO is cumulative: once an operation is removed it cannot re-enter
        # the active search space at a later threshold checkpoint.
        self.mask_reduce = self.mask_reduce.bool() & (rweights >= point)
        self.mask_normal = self.mask_normal.bool() & (nweights >= point)


        for cell in self.cells:
            if cell.reduction:
                mask = self.mask_reduce
                # mask1 = weights >= point
                # self.alphas_reduce = (weights * mask1.float()).requires_grad_()
                # self.alphas_reduce.cuda().requires_grad_()
            else:
                mask = self.mask_normal
                # mask2 = weights >= point
                # self.alphas_normal = (weights * mask2.float()).requires_grad_()
                # self.alphas_normal.cuda().requires_grad_()
            cell.apply_mask(mask)
            # if cell.reduction:
            #     weights = torch.sigmoid(self.alphas_reduce)
            #     mask1 = weights >= point
            #     self.alphas_reduce = weights * mask1.float()
            #     # self.alphas_reduce.cuda().requires_grad_()
            # else:
            #     weights = torch.sigmoid(self.alphas_normal)
            #     mask2 = weights >= point
            #     self.alphas_normal = weights * mask2.float()
            #     # self.alphas_normal.cuda().requires_grad_()

    def set_architecture_state(self, architecture, mask_normal, mask_reduce):
        """Restore architecture tensors and the irreversible KO structure."""
        with torch.no_grad():
            for name in ('alphas_normal', 'alphas_reduce', 'betas_normal', 'betas_reduce'):
                target = getattr(self, name)
                target.copy_(architecture[name].to(target.device))
        self.mask_normal = mask_normal.to(self.alphas_normal.device).bool()
        self.mask_reduce = mask_reduce.to(self.alphas_reduce.device).bool()
        for cell in self.cells:
            cell.apply_mask(self.mask_reduce if cell.reduction else self.mask_normal)

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _loss2(self, input1, target1, input2):
        logits = self(input1)
        return self._criterion2(logits, target1, input2)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        # self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        # self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        # self.betas_normal = Variable(1e-3 * torch.randn(k).cuda(), requires_grad=True)
        # self.betas_reduce = Variable(1e-3 * torch.randn(k).cuda(), requires_grad=True)
        self.alphas_normal = (1e-3 * torch.randn(k, num_ops)).cuda().requires_grad_()
        self.alphas_reduce = (1e-3 * torch.randn(k, num_ops)).cuda().requires_grad_()
        self.betas_normal = (1e-3 * torch.randn(k)).cuda().requires_grad_()
        self.betas_reduce = (1e-3 * torch.randn(k)).cuda().requires_grad_()
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
            self.betas_normal,
            self.betas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights, weights2):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                W2 = weights2[start:end].copy()
                for j in range(n):
                    W[j, :] = W[j, :] * W2[j]
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((PRIMITIVES[k_best], j))

                start = end
                n += 1
            return gene

        n = 3
        start = 2
        weightsr2 = torch.sigmoid(self.betas_reduce[0:2])
        weightsn2 = torch.sigmoid(self.betas_normal[0:2])
        for i in range(self._steps - 1):
            end = start + n
            tw2 = torch.sigmoid(self.betas_reduce[start:end])
            tn2 = torch.sigmoid(self.betas_normal[start:end])
            start = end
            n += 1
            weightsr2 = torch.cat([weightsr2, tw2], dim=0)
            weightsn2 = torch.cat([weightsn2, tn2], dim=0)
        gene_normal = _parse(torch.sigmoid(self.alphas_normal).data.cpu().numpy(), weightsn2.data.cpu().numpy())
        gene_reduce = _parse(torch.sigmoid(self.alphas_reduce).data.cpu().numpy(), weightsr2.data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

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
