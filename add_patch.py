#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module implements some functions to add patches to matplotlib
mashow or to numpy arrays themselves.
"""

import numpy as np

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.colors as color

import imageio


def addpatch(ax, posx, posy, w=1, h=1, color='red'):
    """
    Adds a rectangle patch to a matshow axis.

    Arguments
    ---------
    ax: Axis object
        The axis which contains the current matshow axis.
    posx: int
        The position of the upper left pixel along the x axis
        (the pixel will be contained in the patch).
    posy: int
        The position of the upper left pixel along the y axis
        (the pixel will be contained in the patch).
    w: int
        The with of the patch in pixels.
    h: int
        The height of the patch in pixels.
    color: optional, str
        The patch color. Default is red.
    """
    rect = patches.Rectangle(
        (posx-0.5, posy-0.5),
        h,
        w,
        linewidth=1,
        edgecolor=color,
        facecolor='none')
    ax.add_patch(rect)


def pix_patch(im, pix=(0, 0), c='r'):
    """Paint the im pixel at position pix in color.

    Arguments
    ---------
    im: (m, n) numpy array
        The image: one band or rgba.
    pix: tuple
        Pixel to color (y, x).
    c: str
        Color to use.

    Returns
    -------
    cim: (m, n) numpy array
        Colored image.
    """
    # In case monoband image.
    dyn = 1 if im.max() <= 1 else 255

    # if im.ndim == 2:
    #     im = np.tile(im[:, :, np.newaxis], [1, 1, 4])
    #     im[:, :, 3] = dyn
    cim = im.copy()

    cim[pix[0], pix[1], :] = np.floor(
        dyn*np.asarray(color.to_rgba(c))).astype(int)
    return cim


def rect_patch(im, ulpix=(0, 0), lrpix=(1, 1), c='r'):
    """Paint rectangle patch.

    Arguments
    ---------
    im: (m, n) numpy array
        The image.
    ulpix: tuple
        Upper left pixel to color (y, x).
    lrpix: tuple
        Lower right pixel to color (y, x).
    c: str
        Color to use.

    Returns
    -------
    cim: (m, n) numpy array
        Colored image.
    """
    cim = im.copy()

    val = np.floor(255*np.asarray(color.to_rgba(c))).astype(int)

    # upper and lower edge
    for x in range(ulpix[1], lrpix[1]+1):
        cim[ulpix[0], x, :] = val
        cim[lrpix[0], x, :] = val

    # left and right edge
    for y in range(ulpix[0], lrpix[0]+1):
        cim[y, ulpix[1], :] = val
        cim[y, lrpix[1], :] = val

    return cim
