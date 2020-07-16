# -*- coding: utf-8 -*-

import numpy as np

import inpystem

from inpystem.tools.PCA import PcaHandler


class Data:
    """Class that defines the methods to handle data for the paper.
    """

    def __init__(self, key, th, seed=0):
        """Initialization function
        """
        self.key_DATA = key

        # Getting PCA-thresholded noise-free data. ---------------------
        #

        # Get noise-free acquisition in PCA space.
        nf_acq = inpystem.load_key(
            key, ndim=3,
            dev={'PCA_transform': True, 'PCA_th': th},
            verbose=False
            )

        # Get noise-free acquisition after PCA thresholding in true
        # space
        nf_spim = nf_acq.inverse_transform(nf_acq.data)

        # Adding noise. ------------------------------------------------
        #

        # Set noise levels.
        snr = 26.75
        P = np.mean(nf_spim**2)
        self.sig = np.sqrt(P/10**(snr/10))

        # Draw noise matrix
        if seed is not None:
            np.random.seed(seed)
        noise_mat = np.random.randn(*nf_spim.shape)

        # Construct noised images with sig1 and sig2
        #
        m, n, M = nf_spim.shape

        # Add noise to the data.
        n_spim_tmp = nf_spim + self.sig * noise_mat

        # Perform PCA to noised data
        handler = PcaHandler(n_spim_tmp, PCA_th=th, verbose=False)

        # Get output
        Y_PCA, InfoOut = handler.Y_PCA, handler.InfoOut

        # Store all information.
        self.std = Y_PCA.std()
        self.Y = Y_PCA/Y_PCA.std()
        self.H = InfoOut['H']
        self.Ym = InfoOut['Ym']
        self.d = InfoOut['d']

        # PCA-th noise-free data in full space
        self.X = nf_spim

        # PCA-th noise-free data in PCA-thresholded space
        self.Xy = nf_acq.data

        # Save __init__ arguments
        self.th = th

    def inverse_transform(self, Y_PCA):
        """
        """
        m, n, M = self.X.shape

        Y_PCA = Y_PCA * self.std
        back_data = self.H @ Y_PCA.reshape((m*n, self.th)).T

        return (back_data).T.reshape((m, n, M)) + self.Ym

    def draw_noise(self):

        Data.__init__(self.key_DATA, self.th, None)


class R1(Data):
    """Defines the HR1 data for the paper.
    """

    def __init__(self, th=9):
        """__init__ function.
        """
        self.key = 'R1'
        Data.__init__(self, 'HR-Spim12', th)


class R2(Data):
    """Defines the HR1 data for the paper.
    """

    def __init__(self, th=7):
        """__init__ function.
        """
        self.key = 'R2'
        Data.__init__(self, 'HR-Spim4-2-ali', th)


class Synth(Data):
    """Defines the HR1 data for the paper.
    """

    def __init__(self, th=4):
        """__init__ function.
        """
        self.key = 'S'
        Data.__init__(self, 'HR-Synth', th)


class Real:
    """Class that defines the methods to handle data for the paper.
    """

    def __init__(self):
        """Initialization function
        """
        self.key = 'Real'

        # Get noise-free acquisition in PCA space.
        #
        th = 7
        acq = inpystem.load_key(
            'HR-Spim4-2-ali', ndim=3,
            dev={'PCA_transform': True, 'PCA_th': th},
            verbose=False
            )
        Y_PCA = acq.data

        acq_X = inpystem.load_key(
            'HR-Spim4-2-ali', ndim=3,
            dev={'PCA_transform': False},
            verbose=False
            )
        self.X = acq_X.data

        self.std = Y_PCA.std()
        self.Y = Y_PCA/Y_PCA.std()
        self.H = acq.PCA_info['H']
        self.Ym = acq.PCA_info['Ym']
        self.d = acq.PCA_info['d']
        self.sig = acq.PCA_info['sigma']

        # Save __init__ arguments
        self.th = th

    def inverse_transform(self, Y_PCA):
        """
        """
        m, n, M = self.X.shape

        Y_PCA = Y_PCA * self.std
        back_data = self.H @ Y_PCA.reshape((m*n, self.th)).T

        return (back_data).T.reshape((m, n, M)) + self.Ym

    def draw_noise(self):

        self.__init__(self.PATH, self.th)
