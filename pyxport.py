#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyXport module

verion: 1.0
author: Etienne MONIER (https://github.com/etienne-monier)
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio

import os
import os.path


def _to_im_scale(im):
    """to_im_scale function.

    This function returns
    """
    return np.uint8(np.floor(255*im))


def plot2im(mat, loc, reference=None, cmap='viridis'):
    """plot2im function.

    This function saves the plotted image when using matshow with mat.
    The colormap is cmap.

    The user can define a reference matrix. Its maximum and minimum
    value will be chosen as the the colormap limits. Above (resp. below)
    values will be thresholded to the colormap upper (resp. lower)
    limit.

    Arguments
    ---------
    mat: numpy array
        The data matrix.
    loc: str
        The saving location.
    reference: numpy array
        A reference matrix.
    cmap: optional, str
        The colormap. Default is viridis.
    """

    # Check that matrix is 2D
    if mat.ndim != 2:
        raise ValueError('The mat parameer should be a 2D Numpy array.'
                         'The given array has shape {}.'.format(mat.shape))

    # Check location. If the directory does not exist, create recursively.
    if (os.path.dirname(loc) != "" and not os.path.isdir(
            os.path.dirname(loc))):
        os.makedirs(os.path.dirname(loc))

    # Catch reference matrix if given.
    if np.any(reference is None):
        reference = mat

    # Create the colormaped image matrix.
    cmap = plt.cm.get_cmap(cmap)
    norm = plt.Normalize(vmin=reference.min(), vmax=reference.max())
    image = cmap(norm(mat))

    # Save the image.
    imageio.imwrite(loc, _to_im_scale(image))


def save_dat(data, loc, sep=' '):
    """Save data to .dat file for LaTeX pgfplotstable package usage.

    This function accepts three types of data.

    1. A Numpy array was given. Then, the .dat file will only contain
    its values separated by newline.

    2. A list/typle of one or two Numpy arrays was given. Then, the .dat
    file will have the values of the different arrays separated by the
    optional separator sep.

    3. A dictionary of Numpy arrays was given. Then, the keys will be
    given in the first line of the .dat file.

    Note
    ----
        In the case of multiple arrays input, the arrays should have the
        save size. Otherwise, a ValueError will be raised.

    Arguments
    ---------
    data: 1D array, tuple or list of 1D arrays, dictionary of 1D arrays)
        Arrays to save.
    loc: str
        Place to save the data.
    sep: str
        Data separator. Default is one space.
    """

    # Search data type and prepare data.

    dico_flag = False

    # Is it a single array ?
    if type(data).__module__ == np.__name__:
        if data.ndim > 1:
            raise ValueError(
                'The data was a numpy array of dimension {} where the only'
                'dimension allowed is 1. Give a tuple, list or dictionary'
                'instead.'.format(data.ndim))
        data_out = data[np.newaxis, :]

    # Is it a list or tuple ?
    elif type(data) is tuple or type(data) is list:
        if len(set([type(a).__module__ == np.__name__ for a in data])) != 1:
            raise ValueError(
                'Some elements of the data list were not numpy data.')
        if len(set([a.shape for a in data])) != 1:
            raise ValueError(
                'Elements of the data list have inconsistent shapes.')
        # if len(data)>2:
        #   raise ValueError('A list/tuple of length superior to 2 was
        # given.
        # This is not accepted by pgfplotstable. Use a dictionary
        # instead to add keywords.')
        data_out = np.asarray(data)

    # Is it a dico ?
    elif type(data) is dict:
        if len(set([type(a).__module__ == np.__name__ for a in list(
                data.values())])) != 1:
            raise ValueError(
                'Some elements of the data list were not numpy data.')
        if len(set([a.shape for a in list(data.values())])) != 1:
            raise ValueError(
                'Elements of the data list have inconsistent shapes.')
        dico_flag = True
        data_out = np.stack(tuple(data.values()))

    # Not known data
    else:
        raise ValueError(
            'The data is not a numpy array, nor a tuple, nor a list, nor'
            'a dictionary. Instead, its type is {}'.format(type(data)))

    # Create dir if absent
    if (os.path.dirname(loc) != '' and not os.path.isdir(
            os.path.dirname(loc))):
        os.makedirs(os.path.dirname(loc))

    # Catch data shape
    N, L = data_out.shape

    # Open file
    file = open(loc, 'w')

    # If a dictionary was given, write keys
    if dico_flag:
        for ind, key in enumerate(data.keys()):
            if ind == 0:
                file.write('{}'.format(key))
            else:
                file.write('{}{}'.format(sep, key))
        file.write('\n')

    # Then, the data is written
    for l in range(L):
        for n in range(N):
            if n == 0:
                file.write('{}'.format(data_out[n, l]))
            else:
                file.write('{}{}'.format(sep, data_out[n, l]))
        file.write('\n')

    # Close file
    file.close()
