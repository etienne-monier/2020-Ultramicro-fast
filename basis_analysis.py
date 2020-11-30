#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.fftpack as fftp

import numpy as np
from numpy.linalg import norm

import pywt


def FourierApproxError(X0, r):
    r"""Computes the approximation error in Fourier basis for 3D data.

    Let a ratio r be set.
    A band-by-band Fourier transformation is applied to X0. Then, the
    :math:`\ell_2` norm is applied to each spectrum of it. Last, only
    the r percent higher norms are kept, the other norms spectra are set
    to 0. An inverse band-by-band Fourier transformation is applied and the
    NMSE is computed.

    This is done for all elements in r.

    Arguments
    ---------
    X0: (m, n, L) numpy array
        The data to be studied.
    r: (R, ) numpy array
        The set of ratios to test.

    Returns
    -------
    (R, ) numpy array
        The approximation error for all r.
    """

    m, n, L = X0.shape

    # FFT transform
    A = fftp.fftn(X0, axes=(0, 1)) / np.sqrt(m*n)

    # Get l2-norm along spectra and sort it in decreasing order
    l2norm = norm(np.abs(A), axis=2)
    l2normsorted = l2norm.copy().flatten()
    l2normsorted[::-1].sort()
    # Get threshold depending on r.
    Threshold = l2normsorted[(np.round(r * (n * m - 1))).astype(int)]

    # Threshold decomposition
    l2normup = np.tile(l2norm, [r.size, 1, 1])
    Thresholdup = np.tile(Threshold[:, np.newaxis, np.newaxis], [1, m, n])
    thresh_l2norm = l2normup <= Thresholdup
    mask = np.moveaxis(np.tile(thresh_l2norm, [L, 1, 1, 1]), 0, 1)

    A = np.tile(np.moveaxis(A, -1, 0), [r.size, 1, 1, 1])
    A[mask] = 0

    # Reconstruction
    Xhat = np.real(np.sqrt(m*n) * fftp.ifftn(A, axes=(-2, -1)))

    # Approximation error
    X0up = np.tile(np.moveaxis(X0, -1, 0), [r.size, 1, 1, 1])
    approxError = np.sum((X0up - Xhat)**2, axis=(1, 2, 3)) / \
        np.sum(X0up**2, axis=(1, 2, 3))

    return (approxError, A)


def DctApproxError(X0, r):
    r"""Computes the approximation error in DCT basis for 3D data.

    Let a ratio r be set.
    A band-by-band DCT transformation is applied to X0. Then, the
    :math:`\ell_2` norm is applied to each spectrum of it. Last, only
    the r percent higher norms are kept, the other norms spectra are set
    to 0. An inverse band-by-band DCT transformation is applied and the
    NMSE is computed.

    This is done for all elements in r.

    Arguments
    ---------
    X0: (m, n, L) numpy array
        The data to be studied.
    r: (R, ) numpy array
        The set of ratios to test.

    Returns
    -------
    (R, ) numpy array
        The approximation error for all r.
    """

    m, n, L = X0.shape

    # FFT transform
    A = fftp.dct(X0, axis=0, norm='ortho')
    A = fftp.dct(A, axis=1, norm='ortho')

    # Get l2-norm along spectra and sort it in decreasing order
    l2norm = norm(A, axis=2)
    l2normsorted = l2norm.copy().flatten()
    l2normsorted[::-1].sort()
    # Get threshold depending on r.
    Threshold = l2normsorted[(np.round(r * (n * m - 1))).astype(int)]

    # Threshold decomposition
    l2normup = np.tile(l2norm, [r.size, 1, 1])
    Thresholdup = np.tile(Threshold[:, np.newaxis, np.newaxis], [1, m, n])
    thresh_l2norm = l2normup <= Thresholdup
    mask = np.moveaxis(np.tile(thresh_l2norm, [L, 1, 1, 1]), 0, 1)

    A = np.tile(np.moveaxis(A, -1, 0), [r.size, 1, 1, 1])
    A[mask] = 0

    # Reconstruction
    Xhat = fftp.idct(A, axis=-1, norm='ortho')
    Xhat = fftp.idct(Xhat, axis=-2, norm='ortho')

    # Approximation error
    X0up = np.tile(np.moveaxis(X0, -1, 0), [r.size, 1, 1, 1])
    approxError = np.sum((X0up - Xhat)**2, axis=(1, 2, 3)) / \
        np.sum(X0up**2, axis=(1, 2, 3))

    return (approxError, A)


def WaveletApproxError(X0, r, w):
    r"""Computes the approximation error in wavelet basis for 3D data.

    Let a ratio r be set.
    A band-by-band wavelet transformation is applied to X0. Then, the
    :math:`\ell_2` norm is applied to each spectrum of it. Last, only
    the r percent higher norms are kept, the other norms spectra are set
    to 0. An inverse band-by-band wavelet transformation is applied and the
    NMSE is computed.

    This is done for all elements in r.

    To get the available wavelets, do:

    .. code::python

        >>> import pywt
        >>> pywt.families(short=True)
        # Put short to False to have full names.
        >>> pywt.wavelist(family=db)

    Arguments
    ---------
    X0: (m, n, L) numpy array
        The data to be studied.
    r: (R, ) numpy array
        The set of ratios to test.
    w: str
        The wavelet to use.

    Returns
    -------
    (R, ) numpy array
        The approximation error for all r.
    """

    # FFT transform
    A, coeffs = pywt.coeffs_to_array(
        pywt.wavedec2(X0, w, axes=(0, 1)), axes=(0, 1))

    m, n, L = A.shape

    # Get l2-norm along spectra and sort it in decreasing order
    l2norm = norm(A, axis=2)
    l2normsorted = l2norm.copy().flatten()
    l2normsorted[::-1].sort()
    # Get threshold depending on r.
    Threshold = l2normsorted[(np.round(r * (n * m - 1))).astype(int)]

    # Threshold decomposition
    l2normup = np.tile(l2norm, [r.size, 1, 1])
    Thresholdup = np.tile(Threshold[:, np.newaxis, np.newaxis], [1, m, n])
    thresh_l2norm = l2normup <= Thresholdup
    mask = np.moveaxis(np.tile(thresh_l2norm, [L, 1, 1, 1]), 0, 1)

    A = np.tile(np.moveaxis(A, -1, 0), [r.size, 1, 1, 1])
    A[mask] = 0
    A = np.transpose(A, [2, 3, 1, 0])

    # Reconstruction
    Xhat = np.real(pywt.waverec2(
        pywt.array_to_coeffs(A, coeffs, output_format='wavedec2'),
        w,
        axes=(0, 1)
        )
    )

    # Approximation error
    X0up = np.tile(np.moveaxis(X0, -1, 0), [r.size, 1, 1, 1])
    Xhat = np.transpose(Xhat, [3, 2, 0, 1])
    approxError = np.sum((X0up - Xhat)**2, axis=(1, 2, 3)) / \
        np.sum(X0up**2, axis=(1, 2, 3))

    return (approxError, A)
