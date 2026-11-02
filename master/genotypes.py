from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# PRIMITIVES = [
#     # 'MBConv_3x3x3',
#     # 'none',
#     'MBConv_5x5x5_3',
#     'MBConv_5x5x5_6',
#     'max_pool_3x3x3',
#     'avg_pool_3x3x3',
#     'skip_connect',
#     'sep_conv_3x3x3',
#     'sep_conv_5x5x5',
#     'dil_conv_3x3x3',
#     'dil_conv_5x5x5'
# ]

PRIMITIVES = [
    'skip_connect',
    'Block_16_2x2x2_12',
    'Block_16_4x4x4_12',
    # 'Block_16_8x8x8_12',
    # 'Block_32_2x2x2_12',
    # 'Block_32_4x4x4_12',
    # 'Block_32_8x8x8_12',

    'Block_16_2x2x2_4',
    'Block_16_4x4x4_4',
    # 'Block_16_8x8x8_4',
    # 'Block_32_2x2x2_4',
    # 'Block_32_4x4x4_4',
    # 'Block_32_8x8x8_4',

    'Avg_pool_3x3x3',
    'Max_pool_3x3x3',

    # 'SiLU',
    # 'ReLU',
    # 'ReLU6',
]



# PRIMITIVES_2 = [
#     'none',
#     'max_pool_3x3',
#     'avg_pool_3x3',
#     'skip_connect',
#     'sep_conv_3x3',
#     'sep_conv_5x5',
#     'dil_conv_3x3',
#     'dil_conv_5x5'
# ]

# NASNet = Genotype(
#   normal = [
#     ('sep_conv_5x5', 1),
#     ('sep_conv_3x3', 0),
#     ('sep_conv_5x5', 0),
#     ('sep_conv_3x3', 0),
#     ('avg_pool_3x3', 1),
#     ('skip_connect', 0),
#     ('avg_pool_3x3', 0),
#     ('avg_pool_3x3', 0),
#     ('sep_conv_3x3', 1),
#     ('skip_connect', 1),
#   ],
#   normal_concat = [2, 3, 4, 5, 6],
#   reduce = [
#     ('sep_conv_5x5', 1),
#     ('sep_conv_7x7', 0),
#     ('max_pool_3x3', 1),
#     ('sep_conv_7x7', 0),
#     ('avg_pool_3x3', 1),
#     ('sep_conv_5x5', 0),
#     ('skip_connect', 3),
#     ('avg_pool_3x3', 2),
#     ('sep_conv_3x3', 2),
#     ('max_pool_3x3', 1),
#   ],
#   reduce_concat = [4, 5, 6],
# )
#
# AmoebaNet = Genotype(
#   normal = [
#     ('avg_pool_3x3', 0),
#     ('max_pool_3x3', 1),
#     ('sep_conv_3x3', 0),
#     ('sep_conv_5x5', 2),
#     ('sep_conv_3x3', 0),
#     ('avg_pool_3x3', 3),
#     ('sep_conv_3x3', 1),
#     ('skip_connect', 1),
#     ('skip_connect', 0),
#     ('avg_pool_3x3', 1),
#     ],
#   normal_concat = [4, 5, 6],
#   reduce = [
#     ('avg_pool_3x3', 0),
#     ('sep_conv_3x3', 1),
#     ('max_pool_3x3', 0),
#     ('sep_conv_7x7', 2),
#     ('sep_conv_7x7', 0),
#     ('avg_pool_3x3', 1),
#     ('max_pool_3x3', 0),
#     ('max_pool_3x3', 1),
#     ('conv_7x1_1x7', 0),
#     ('sep_conv_3x3', 5),
#   ],
#   reduce_concat = [3, 4, 6]
# )


DARTS_MAMBA_1 = Genotype(normal=[('VSSBlock_32_4x4_12', 1), ('VSSBlock_32_2x2_12', 0), ('VSSBlock_32_2x2_12', 2), ('VSSBlock_32_2x2_12', 1), ('VSSBlock_32_2x2_12', 0), ('skip_connect', 2), ('VSSBlock_32_2x2_12', 4), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 3), ('VSSBlock_32_4x4_12', 2), ('skip_connect', 0)], reduce_concat=range(2, 6))

DARTS_MAMBA_2 = Genotype(normal=[('VSSBlock_32_2x2_4', 1), ('VSSBlock_32_2x2_4', 0), ('VSSBlock_32_2x2_12', 2), ('VSSBlock_32_4x4_12', 1), ('VSSBlock_32_2x2_4', 0), ('VSSBlock_32_2x2_12', 1), ('VSSBlock_32_2x2_12', 3), ('VSSBlock_32_2x2_12', 4)], normal_concat=range(2, 6), reduce=[('VSSBlock_32_4x4_4', 0), ('VSSBlock_32_4x4_4', 1), ('VSSBlock_32_4x4_12', 0), ('VSSBlock_32_4x4_12', 1), ('VSSBlock_32_4x4_4', 0), ('VSSBlock_32_4x4_12', 2), ('VSSBlock_32_4x4_4', 2), ('VSSBlock_32_4x4_4', 0)], reduce_concat=range(2, 6))

DARTS_MAMBA_3_100_6L = Genotype(normal=[('VSSBlock_32_4x4_4', 0), ('VSSBlock_32_4x4_4', 1), ('VSSBlock_32_4x4_12', 2), ('VSSBlock_32_4x4_12', 0), ('VSSBlock_32_4x4_12', 1), ('VSSBlock_32_4x4_4', 2), ('VSSBlock_32_2x2_12', 3), ('VSSBlock_32_4x4_4', 0)], normal_concat=range(2, 6), reduce=[('VSSBlock_32_2x2_12', 0), ('VSSBlock_32_2x2_12', 1), ('skip_connect', 2), ('VSSBlock_32_4x4_12', 0), ('VSSBlock_32_2x2_4', 1), ('VSSBlock_32_4x4_12', 0), ('VSSBlock_32_2x2_12', 2), ('VSSBlock_32_2x2_12', 4)], reduce_concat=range(2, 6))

DARTS_MAMBA4 = Genotype(normal=[('VSSBlock_16_4x4_4', 0), ('VSSBlock_16_4x4_4', 1), ('VSSBlock_16_4x4_4', 0), ('VSSBlock_16_4x4_4', 1), ('skip_connect', 3), ('VSSBlock_16_4x4_12', 0), ('skip_connect', 3), ('VSSBlock_16_4x4_12', 4)], normal_concat=range(2, 6), reduce=[('VSSBlock_16_2x2_4', 1), ('VSSBlock_16_4x4_12', 0), ('VSSBlock_16_4x4_12', 1), ('VSSBlock_16_4x4_4', 2), ('VSSBlock_16_4x4_4', 1), ('VSSBlock_16_4x4_4', 2), ('VSSBlock_16_4x4_4', 0), ('VSSBlock_16_4x4_12', 4)], reduce_concat=range(2, 6))

DARTS_MAMBA5_92 = Genotype(normal=[('Block_16_2x2_4', 0), ('Block_16_4x4_4', 1), ('Block_16_2x2_12', 0), ('avg_pool_3x3x3', 2), ('Block_16_4x4_4', 3), ('skip_connect', 0), ('skip_connect', 2), ('Block_16_2x2_12', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3x3', 1), ('Block_16_4x4_4', 0), ('Block_16_2x2_4', 1), ('max_pool_3x3x3', 2), ('Block_16_2x2_12', 1), ('max_pool_3x3x3', 3), ('max_pool_3x3x3', 1), ('Block_16_4x4_12', 4)], reduce_concat=range(2, 6))

DARTS_MAMBA6 = Genotype(normal=[('Block_16_2x2_12', 0), ('Block_16_4x4_4', 1), ('skip_connect', 0), ('max_pool_3x3x3', 2), ('max_pool_3x3x3', 2), ('skip_connect', 3), ('skip_connect', 2), ('Block_16_2x2_12', 0)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('skip_connect', 0), ('Block_16_2x2_4', 1), ('Block_16_2x2_4', 2), ('Block_16_2x2_4', 1), ('Block_16_4x4_12', 2), ('max_pool_3x3x3', 1), ('Block_16_4x4_4', 4)], reduce_concat=range(2, 6))


DARTS_MAMBA7 = Genotype(normal=[('avg_pool_3x3x3', 0), ('avg_pool_3x3x3', 1), ('Block_16_2x2x2_12', 0), ('avg_pool_3x3x3', 2), ('avg_pool_3x3x3', 2), ('avg_pool_3x3x3', 3), ('avg_pool_3x3x3', 2), ('Block_16_4x4x4_12', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3x3', 1), ('skip_connect', 0), ('max_pool_3x3x3', 1), ('max_pool_3x3x3', 2), ('max_pool_3x3x3', 1), ('max_pool_3x3x3', 3), ('Block_16_2x2x2_12', 1), ('Block_16_4x4x4_12', 4)], reduce_concat=range(2, 6))

DARTS_MAMBA8 = Genotype(normal=[('avg_pool_3x3x3', 0), ('Block_16_4x4x4_12', 1), ('skip_connect', 0), ('Block_16_4x4x4_12', 1), ('Block_16_4x4x4_12', 0), ('Block_16_4x4x4_4', 3), ('Block_16_4x4x4_12', 0), ('Block_16_4x4x4_12', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3x3', 1), ('Block_16_2x2x2_4', 0), ('Block_16_2x2x2_4', 1), ('Block_16_4x4x4_4', 2), ('Block_16_4x4x4_12', 2), ('avg_pool_3x3x3', 3), ('Block_16_4x4x4_12', 0), ('Block_16_2x2x2_4', 1)], reduce_concat=range(2, 6))

DARTS_MAMBA9 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('Block_16_4x4x4_4', 2), ('avg_pool_3x3x3', 0), ('Block_16_2x2x2_12', 1), ('avg_pool_3x3x3', 0), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('Block_16_2x2x2_12', 0), ('Block_16_2x2x2_12', 1), ('avg_pool_3x3x3', 0), ('Block_16_4x4x4_4', 1), ('avg_pool_3x3x3', 0), ('Block_16_4x4x4_4', 2), ('Block_16_4x4x4_4', 2), ('Block_16_4x4x4_4', 3)], reduce_concat=range(2, 6))

DARTS_MAMBA10 = Genotype(normal=[('skip_connect', 0), ('max_pool_3x3x3', 1), ('Block_16_4x4x4_12', 2), ('skip_connect', 0), ('Block_16_2x2x2_12', 0), ('Block_16_4x4x4_4', 2), ('Block_16_4x4x4_12', 0), ('Block_16_2x2x2_12', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('max_pool_3x3x3', 2), ('Block_16_2x2x2_4', 1), ('Block_16_4x4x4_12', 3), ('skip_connect', 1), ('Block_16_4x4x4_4', 4)], reduce_concat=range(2, 6))

DARTS_MAMBA11 = Genotype(normal=[('max_pool_3x3x3', 0), ('max_pool_3x3x3', 1), ('Block_16_2x2x2_4', 0), ('max_pool_3x3x3', 2), ('Block_16_2x2x2_12', 3), ('max_pool_3x3x3', 2), ('max_pool_3x3x3', 4), ('max_pool_3x3x3', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3x3', 0), ('avg_pool_3x3x3', 1), ('Block_16_2x2x2_4', 2), ('avg_pool_3x3x3', 0), ('skip_connect', 3), ('avg_pool_3x3x3', 0), ('skip_connect', 3), ('skip_connect', 4)], reduce_concat=range(2, 6))


DARTS_MAMBA12 = Genotype(normal=[('Block_16_4x4x4_4', 1), ('max_pool_3x3x3', 0), ('Block_16_2x2x2_4', 2), ('max_pool_3x3x3', 1), ('Block_16_2x2x2_4', 1), ('Block_16_2x2x2_12', 2), ('skip_connect', 1), ('Block_16_4x4x4_4', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3x3', 0), ('skip_connect', 1), ('max_pool_3x3x3', 0), ('Block_16_2x2x2_12', 2), ('Block_16_2x2x2_4', 2), ('Block_16_2x2x2_4', 3), ('skip_connect', 2), ('Block_16_2x2x2_12', 3)], reduce_concat=range(2, 6))


DARTS_MAMBA13 = Genotype(normal=[('Block_16_2x2x2_4', 1), ('Block_16_2x2x2_12', 0), ('Block_16_2x2x2_12', 2), ('Block_16_2x2x2_12', 1), ('Block_16_2x2x2_12', 1), ('Block_16_2x2x2_12', 3), ('Avg_pool_3x3x3', 1), ('Avg_pool_3x3x3', 2)], normal_concat=range(2, 6), reduce=[('Block_16_2x2x2_4', 1), ('Max_pool_3x3x3', 0), ('Block_16_2x2x2_4', 2), ('Max_pool_3x3x3', 0), ('Block_16_2x2x2_12', 2), ('Max_pool_3x3x3', 3), ('Max_pool_3x3x3', 3), ('Max_pool_3x3x3', 4)], reduce_concat=range(2, 6))


DARTS_MAMBA14 = Genotype(normal=[('Block_16_2x2x2_12', 1), ('Block_16_4x4x4_12', 0), ('Block_16_2x2x2_12', 2), ('Block_16_2x2x2_12', 1), ('Block_16_2x2x2_12', 1), ('Block_16_2x2x2_4', 2), ('Block_16_2x2x2_4', 1), ('Block_16_4x4x4_12', 0)], normal_concat=range(2, 6), reduce=[('Block_16_2x2x2_12', 0), ('Block_16_2x2x2_12', 1), ('Block_16_2x2x2_4', 0), ('Block_16_2x2x2_12', 2), ('Block_16_2x2x2_4', 3), ('Block_16_2x2x2_4', 0), ('Block_16_4x4x4_12', 3), ('Max_pool_3x3x3', 4)], reduce_concat=range(2, 6))



DARTS_MAMBA15 = Genotype(normal=[('Block_16_2x2x2_4', 1), ('Max_pool_3x3x3', 0), ('Block_16_2x2x2_12', 2), ('Block_16_2x2x2_12', 1), ('Block_16_4x4x4_12', 1), ('Block_16_2x2x2_4', 2), ('Block_16_4x4x4_4', 3), ('Block_16_2x2x2_12', 1)], normal_concat=range(2, 6), reduce=[('Block_16_2x2x2_12', 0), ('Block_16_2x2x2_4', 1), ('Block_16_2x2x2_4', 2), ('Block_16_2x2x2_4', 0), ('Block_16_2x2x2_12', 3), ('Block_16_2x2x2_12', 1), ('Max_pool_3x3x3', 2), ('Max_pool_3x3x3', 4)], reduce_concat=range(2, 6))


DARTS_MAMBA16 = Genotype(normal=[('Block_16_2x2x2_12', 1), ('Block_16_2x2x2_12', 0), ('Block_16_2x2x2_12', 0), ('Block_16_2x2x2_4', 1), ('Block_16_2x2x2_4', 2), ('Block_16_2x2x2_4', 1), ('Block_16_2x2x2_12', 1), ('Block_16_2x2x2_4', 2)], normal_concat=range(2, 6), reduce=[('Block_16_2x2x2_12', 0), ('Block_16_2x2x2_12', 1), ('Block_16_2x2x2_4', 2), ('Block_16_2x2x2_4', 1), ('Block_16_4x4x4_12', 2), ('Max_pool_3x3x3', 3), ('Block_16_4x4x4_12', 4), ('Max_pool_3x3x3', 3)], reduce_concat=range(2, 6))
DARTS_MAMBA17 = Genotype(normal=[('Block_16_2x2x2_12', 1), ('Block_16_2x2x2_4', 0), ('Block_16_2x2x2_12', 2), ('Block_16_2x2x2_12', 1), ('Block_16_2x2x2_12', 3), ('Block_16_2x2x2_4', 2), ('Block_16_2x2x2_12', 1), ('Block_16_4x4x4_12', 3)], normal_concat=range(2, 6), reduce=[('Block_16_2x2x2_12', 0), ('Block_16_2x2x2_12', 1), ('Block_16_2x2x2_12', 1), ('Block_16_2x2x2_4', 2), ('Block_16_2x2x2_12', 3), ('Block_16_2x2x2_12', 1), ('Max_pool_3x3x3', 3), ('Max_pool_3x3x3', 4)], reduce_concat=range(2, 6))


DARTS_MAMBA18 = Genotype(normal=[('Block_16_4x4x4_4', 1), ('Block_16_4x4x4_12', 0), ('Block_16_4x4x4_4', 2), ('Block_16_2x2x2_12', 1), ('Block_16_4x4x4_12', 2), ('Block_16_4x4x4_4', 3), ('Block_16_4x4x4_4', 4), ('Block_16_4x4x4_4', 1)], normal_concat=range(2, 6), reduce=[('Max_pool_3x3x3', 0), ('Block_16_2x2x2_4', 1), ('Block_16_4x4x4_12', 0), ('Block_16_2x2x2_12', 2), ('Block_16_4x4x4_4', 1), ('Block_16_4x4x4_4', 3), ('Max_pool_3x3x3', 3), ('Max_pool_3x3x3', 2)], reduce_concat=range(2, 6))
DARTS_MAMBA19 = Genotype(normal=[('Block_16_4x4x4_12', 0), ('Block_16_4x4x4_4', 1), ('Block_16_4x4x4_4', 2), ('Block_16_2x2x2_12', 1), ('Block_16_4x4x4_12', 2), ('Block_16_4x4x4_4', 3), ('Block_16_4x4x4_4', 4), ('Block_16_4x4x4_12', 2)], normal_concat=range(2, 6), reduce=[('Max_pool_3x3x3', 0), ('Block_16_2x2x2_4', 1), ('Block_16_2x2x2_12', 2), ('Block_16_4x4x4_12', 0), ('Block_16_4x4x4_4', 1), ('Block_16_2x2x2_4', 3), ('Max_pool_3x3x3', 3), ('Max_pool_3x3x3', 2)], reduce_concat=range(2, 6))
DARTS_MAMBA20 = Genotype(normal=[('Block_16_2x2x2_4', 1), ('Block_16_4x4x4_4', 0), ('Block_16_4x4x4_12', 1), ('Block_16_2x2x2_4', 2), ('Block_16_2x2x2_12', 2), ('Block_16_4x4x4_12', 3), ('Block_16_2x2x2_12', 1), ('Block_16_4x4x4_12', 4)], normal_concat=range(2, 6), reduce=[('Block_16_2x2x2_12', 1), ('Max_pool_3x3x3', 0), ('Block_16_2x2x2_12', 2), ('Block_16_2x2x2_12', 1), ('Block_16_2x2x2_12', 3), ('Block_16_2x2x2_4', 2), ('skip_connect', 4), ('Block_16_2x2x2_12', 3)], reduce_concat=range(2, 6))
DARTS_MAMBA21 = Genotype(normal=[('Block_16_2x2x2_12', 1), ('Block_16_4x4x4_12', 0), ('Block_16_2x2x2_4', 2), ('Block_16_2x2x2_4', 1), ('Block_16_2x2x2_4', 2), ('Block_16_4x4x4_12', 1), ('Block_16_4x4x4_12', 4), ('Block_16_4x4x4_12', 3)], normal_concat=range(2, 6), reduce=[('Max_pool_3x3x3', 1), ('Max_pool_3x3x3', 0), ('Block_16_2x2x2_12', 2), ('Max_pool_3x3x3', 1), ('Max_pool_3x3x3', 2), ('Block_16_2x2x2_12', 3), ('Max_pool_3x3x3', 1), ('Max_pool_3x3x3', 2)], reduce_concat=range(2, 6))


DARTS_MAMBA22 = Genotype(normal=[('Block_16_4x4x4_12', 1), ('Block_16_2x2x2_12', 0), ('Block_16_4x4x4_12', 2), ('Block_16_4x4x4_4', 1), ('Block_16_4x4x4_12', 1), ('Block_16_4x4x4_4', 3), ('Block_16_4x4x4_4', 2), ('Block_16_4x4x4_12', 1)], normal_concat=range(2, 6), reduce=[('Max_pool_3x3x3', 1), ('Max_pool_3x3x3', 0), ('Block_16_4x4x4_4', 2), ('Max_pool_3x3x3', 1), ('skip_connect', 3), ('Max_pool_3x3x3', 2), ('Block_16_4x4x4_12', 3), ('Avg_pool_3x3x3', 1)], reduce_concat=range(2, 6))


DARTS_MAMBA23 = Genotype(normal=[('Block_16_2x2x2_4', 1), ('Block_16_4x4x4_12', 0), ('Block_16_4x4x4_4', 2), ('Block_16_4x4x4_12', 1), ('Block_16_4x4x4_12', 1), ('Block_16_4x4x4_4', 3), ('Block_16_4x4x4_4', 2), ('Block_16_4x4x4_12', 3)], normal_concat=range(2, 6), reduce=[('Max_pool_3x3x3', 0), ('Max_pool_3x3x3', 1), ('Block_16_4x4x4_4', 0), ('Max_pool_3x3x3', 2), ('Block_16_4x4x4_12', 2), ('Block_16_2x2x2_12', 3), ('Max_pool_3x3x3', 3), ('Block_16_2x2x2_12', 4)], reduce_concat=range(2, 6))


DARTS_MAMBA24 = Genotype(normal=[('Block_16_2x2x2_12', 1), ('Block_16_4x4x4_12', 0), ('Block_16_2x2x2_4', 1), ('Block_16_4x4x4_12', 2), ('Block_16_4x4x4_12', 1), ('Block_16_4x4x4_4', 2), ('Block_16_4x4x4_4', 2), ('Block_16_2x2x2_4', 1)], normal_concat=range(2, 6), reduce=[('Max_pool_3x3x3', 0), ('Block_16_4x4x4_12', 1), ('Block_16_4x4x4_4', 0), ('Block_16_4x4x4_12', 1), ('Block_16_4x4x4_4', 3), ('Block_16_4x4x4_4', 1), ('Block_16_4x4x4_4', 3), ('Block_16_4x4x4_4', 4)], reduce_concat=range(2, 6))


DARTS_MAMBA25 = Genotype(normal=[('skip_connect', 1), ('Block_16_4x4x4_12', 0), ('Block_16_2x2x2_12', 2), ('skip_connect', 1), ('Block_16_2x2x2_12', 2), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('Block_16_2x2x2_4', 0), ('Block_16_2x2x2_4', 1), ('Block_16_2x2x2_12', 2), ('Block_16_2x2x2_12', 1), ('Max_pool_3x3x3', 3), ('Max_pool_3x3x3', 2), ('Max_pool_3x3x3', 3), ('Block_16_2x2x2_12', 4)], reduce_concat=range(2, 6))

DARTS_MAMBA = DARTS_MAMBA25

em_dim_32_96 = Genotype(normal=[('Block_16_2x2x2_12', 1), ('Block_16_4x4x4_12', 0), ('Block_16_2x2x2_4', 1), ('Block_16_4x4x4_12', 2), ('Block_16_4x4x4_12', 1), ('Block_16_4x4x4_4', 2), ('Block_16_4x4x4_4', 2), ('Block_16_2x2x2_4', 1)], normal_concat=range(2, 6), reduce=[('Max_pool_3x3x3', 0), ('Block_16_4x4x4_12', 1), ('Block_16_4x4x4_4', 0), ('Block_16_4x4x4_12', 1), ('Block_16_4x4x4_4', 3), ('Block_16_4x4x4_4', 1), ('Block_16_4x4x4_4', 3), ('Block_16_4x4x4_4', 4)], reduce_concat=range(2, 6))
# DARTS_MAMBA = em_dim_32_96

Sample2_Nknockout = Genotype(normal=[('skip_connect', 1), ('Block_16_4x4x4_12', 0), ('skip_connect', 1), ('Block_16_4x4x4_4', 0), ('Block_16_4x4x4_12', 2), ('Max_pool_3x3x3', 1), ('Max_pool_3x3x3', 1), ('Block_16_2x2x2_4', 4)], normal_concat=range(2, 6), reduce=[('Max_pool_3x3x3', 0), ('Block_16_4x4x4_4', 1), ('Block_16_2x2x2_12', 2), ('Block_16_2x2x2_12', 0), ('Max_pool_3x3x3', 3), ('Max_pool_3x3x3', 2), ('Max_pool_3x3x3', 3), ('Max_pool_3x3x3', 4)], reduce_concat=range(2, 6))
# DARTS_MAMBA = Sample2_Nknockout
Sample2_knockout = Genotype(normal=[('skip_connect', 1), ('Block_16_4x4x4_12', 0), ('Block_16_2x2x2_4', 1), ('Block_16_4x4x4_4', 0), ('Block_16_2x2x2_4', 1), ('Block_16_4x4x4_12', 2), ('Avg_pool_3x3x3', 3), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('Max_pool_3x3x3', 1), ('Max_pool_3x3x3', 0), ('Max_pool_3x3x3', 2), ('skip_connect', 1), ('Max_pool_3x3x3', 3), ('Max_pool_3x3x3', 2), ('Max_pool_3x3x3', 3), ('Max_pool_3x3x3', 2)], reduce_concat=range(2, 6))
# DARTS_MAMBA = Sample2_knockout

# F_SNF_Sample2_Nknockout = Genotype(normal=[('Block_16_4x4x4_4', 1), ('Block_16_4x4x4_4', 0), ('Block_16_2x2x2_4', 2), ('Block_16_4x4x4_12', 1), ('Block_16_4x4x4_4', 1), ('Block_16_4x4x4_4', 3), ('skip_connect', 1), ('Block_16_4x4x4_4', 3)], normal_concat=range(2, 6), reduce=[('Block_16_4x4x4_12', 0), ('Block_16_4x4x4_4', 1), ('Block_16_4x4x4_12', 1), ('Block_16_4x4x4_12', 0), ('Block_16_2x2x2_12', 3), ('Block_16_4x4x4_12', 0), ('Block_16_4x4x4_4', 4), ('Block_16_4x4x4_4', 2)], reduce_concat=range(2, 6))
F_SNF_Sample2_Nknockout = Genotype(normal=[('Block_16_4x4x4_4', 1), ('Block_16_4x4x4_4', 0), ('Block_16_2x2x2_4', 2), ('Block_16_4x4x4_12', 0), ('Block_16_2x2x2_12', 3), ('Block_16_4x4x4_4', 1), ('Avg_pool_3x3x3', 1), ('Max_pool_3x3x3', 4)], normal_concat=range(2, 6), reduce=[('Max_pool_3x3x3', 0), ('Block_16_4x4x4_4', 1), ('Max_pool_3x3x3', 0), ('Block_16_4x4x4_12', 1), ('Block_16_4x4x4_4', 3), ('Max_pool_3x3x3', 0), ('Block_16_4x4x4_12', 4), ('Block_16_2x2x2_12', 2)], reduce_concat=range(2, 6))


# DARTS_MAMBA = F_SNF_Sample2_Nknockout

