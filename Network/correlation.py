# Copyright (c) 2020 Carnegie Mellon University, Wenshan Wang <wenshanw@andrew.cmu.edu>
# For License information please see the LICENSE file in the root directory.

import torch
import torch.nn.functional as F


def FunctionCorrelation(tenFirst, tenSecond, md=4):
    """
    Compute the cost volume between two feature maps using pure PyTorch.

    For each displacement (s2o, s2p) where s2o, s2p ∈ [-md, md],
    the correlation at position (x, y) is:
        sum_c(tenFirst[x, y, c] * tenSecond[x+s2o, y+s2p, c]) / C

    tenFirst:  [B, C, H, W]
    tenSecond: [B, C, H, W]

    Returns: [B, (2*md+1)^2, H, W]
    """
    B, C, H, W = tenFirst.shape

    # Pad tenSecond to handle boundary displacements
    tenSecond_padded = F.pad(tenSecond, (md, md, md, md))

    outputs = []
    for s2p in range(-md, md + 1):   # y displacement (slower)
        for s2o in range(-md, md + 1):  # x displacement (faster)
            shifted = tenSecond_padded[
                :, :, md + s2p: md + s2p + H, md + s2o: md + s2o + W
            ]
            corr = (tenFirst * shifted).sum(dim=1, keepdim=True) / C
            outputs.append(corr)

    return torch.cat(outputs, dim=1)  # [B, (2*md+1)^2, H, W]
