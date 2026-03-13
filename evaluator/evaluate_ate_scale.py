#!/usr/bin/python
# Modified by Wenshan Wang
# Modified by Raul Mur-Artal
# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.

"""
Compute the absolute trajectory error (ATE) with optional scale alignment.
"""

import numpy


def align(model, data, calc_scale=False):
    """Align two trajectories using Horn's closed-form method.

    model: first trajectory  (3 x n)
    data:  second trajectory (3 x n)

    Returns: rot (3x3), trans (3x1), trans_error (1xn), scale
    """
    numpy.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered  = data  - data.mean(1)

    W = numpy.zeros((3, 3))
    for column in range(model.shape[1]):
        W += numpy.outer(model_zerocentered[:, column],
                         data_zerocentered[:, column])
    U, d, Vh = numpy.linalg.svd(W.transpose())
    S = numpy.matrix(numpy.identity(3))
    if numpy.linalg.det(U) * numpy.linalg.det(Vh) < 0:
        S[2, 2] = -1
    rot = U * S * Vh

    if calc_scale:
        rotmodel = rot * model_zerocentered
        dots = 0.0
        norms = 0.0
        for column in range(data_zerocentered.shape[1]):
            dots  += numpy.dot(data_zerocentered[:, column].transpose(),
                               rotmodel[:, column])
            normi  = numpy.linalg.norm(model_zerocentered[:, column])
            norms += normi * normi
        s = float(norms / dots)
    else:
        s = 1.0

    trans = s * data.mean(1) - rot * model.mean(1)
    model_aligned = rot * model + trans
    data_aligned  = s * data
    alignment_error = model_aligned - data_aligned

    trans_error = numpy.sqrt(
        numpy.sum(numpy.multiply(alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error, s


def plot_traj(ax, stamps, traj, style, color, label):
    """Plot a trajectory using matplotlib."""
    stamps.sort()
    interval = numpy.median([s - t for s, t in zip(stamps[1:], stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i] - last < 2 * interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
        elif len(x) > 0:
            ax.plot(x, y, style, color=color, label=label)
            label = ""
            x = []
            y = []
        last = stamps[i]
    if len(x) > 0:
        ax.plot(x, y, style, color=color, label=label)
