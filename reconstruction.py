#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib

import numpy as np
import hyperspy.api as hs

import inquirer
import blessings

import inpystem

import pydata


def inpaint_data(
        data_label, data_ind, method_label, method_ind, DL_verb):
    """Generate the reconstructed data.

    Arguments
    ---------
    data_label: list
        The labels for the data to use.
    data_ind: list
        The index of data to use.
    method_label: list
        The labels for methods to use.
    method_ind: list
        The index of methods to use.
    """
    t = blessings.Terminal()

    # Load data
    Data_obj = []
    if 0 in data_ind:
        Data_obj.append(pydata.R1())
    if 1 in data_ind:
        Data_obj.append(pydata.R2())
    if 2 in data_ind:
        Data_obj.append(pydata.Synth())
    if 3 in data_ind:
        Data_obj.append(pydata.Real())

    # Params
    pix = 0.2

    # 3S parameters
    Lambda_3S = 0

    # FS parameters
    Lambda_CLS = {'R1': 0.0616,
                  'R2': 0.1213,
                  'S': 0.00792,
                  'Real': 0.00792}

    # ITKrMM and wKSVD params.
    P = 25   # Patchsize.
    S = 18   # Sparsity level.
    K = 128  # Dictionnary size.
    L = 1    # Num. of low-rank component to estimate.

    # BPFA
    P_BPFA = 41  # Patch size

    # Methods
    #
    available_methods = [
        'interpolation',
        '3S',
        'CLS',
        'ITKrMM_matlab',
        'wKSVD_matlab',
        'BPFA_matlab']
    methods = [available_methods[pos] for pos in method_ind]

    #
    #
    # Setting
    for cnt, obj in enumerate(Data_obj):

        print('[{}] Data {}'.format(
            t.cyan + '*' + t.normal,
            data_label[cnt]))

        # Prepare data
        #

        # Create mask
        m, n, B = obj.X.shape
        scan = inpystem.Scan.random((m, n), ratio=pix, seed=0)

        # Create inpystem object
        hsdata = hs.signals.Signal1D(obj.Y)
        spim = inpystem.Stem3D(hsdata, scan, verbose=False)

        # Initialization
        np.random.seed(0)
        LS_init = np.random.rand(*obj.Y.shape)
        DL_init = np.random.rand(P**2*obj.Y.shape[-1], K)

        for cnt_m, method in enumerate(methods):

            print('\t[{}] Method {}'.format(
                t.yellow + 'o' + t.normal,
                method_label[cnt_m]))

            # Prepare parameters.
            #
            if method == 'interpolation':
                parameters = {}

            elif method == 'CLS':
                parameters = {
                    'Lambda': Lambda_CLS[obj.key], 'init': LS_init,
                    'Nit': 200 if obj.key == 'S' else None,
                    }

            elif method == '3S':
                parameters = {
                    'Lambda': Lambda_3S,
                    'PCA_info': {'d': obj.d/obj.std, 'sigma': obj.sig/obj.std},
                    'init': LS_init
                }

            elif method.startswith('ITKrMM') or method.startswith('wKSVD'):
                parameters = {
                    'P': P, 'S': S, 'K': K, 'L': L,
                    'init_lr': DL_init[:, :L], 'init': DL_init[:, L:]}

            elif method == 'BPFA_matlab':
                parameters = {'P': P_BPFA}

            else:
                raise ValueError('Unknown method.')

            # Setting data verbose if DL
            if DL_verb and (
                    method.startswith('ITKrMM') or method == 'BPFA_matlab'):
                spim.verbose = True

            # Performing reconstruction
            #
            Shat, InfoOut = spim.restore(
                method, parameters, PCA_transform=False)

            # Setting data verbose if DL
            spim.verbose = False

            # Output handling
            #

            # Perform inverse transform.
            Xhat = obj.inverse_transform(Shat.data)

            dico = {'xhat': Xhat, 'time': InfoOut['time']}
            if 'E' in InfoOut:
                dico['E'] = InfoOut['E']

            # Saving output
            file_dir = pathlib.Path(__file__).parent / 'reconstruction'
            file_dir.mkdir(exist_ok=True)

            file = file_dir / '{}_{}.npz'.format(obj.key, method_label[cnt_m])
            np.savez(file, **dico)


def non_empty_validation(answers, current):
    """Returns False if no answer was given in Checkbox question.

    Arguments
    ---------
    answers: dict
        Previous questions answers.
    current: list
        Current question aswers.

    Returns
    -------
    bool
        True if any answer was given, False otherwise.
    """
    if len(current) == 0:
        raise inquirer.errors.ValidationError(
            '',
            reason='Please select at least one entry.')
    return True


def DL_ignore(answers):
    """Return False if any dictionary-learning method was selected.

    Arguments
    ---------
    answers: dict
        Previous questions answers.

    Returns
    -------
    bool
        True if DL verbosity question should be ignored.
    """
    expected = ['ITKrMM', 'wKSVD', 'BPFA']
    flag = [meth in answers['method'] for meth in expected]
    return True not in flag


if __name__ == '__main__':

    # Sets the inpystem data path.
    DATA_bak = inpystem.read_data_path()
    DATA = pathlib.Path(__file__).parent / 'acquisitions'
    flag = inpystem.set_data_path(DATA)

    if flag is False:
        raise Exception('The data path has not correctly been set.')

    # Perform reconstruction
    try:

        # Options for questions.
        data_choices = ['R1 (Synth)', 'R2 (Synth)', 'S (Synth)', 'R2 (Real)']
        method_choices = ['NN', '3S', 'CLS', 'ITKrMM', 'wKSVD', 'BPFA']

        # Questions to select the data and methods.
        questions = [
            inquirer.Checkbox(
                'data',
                message="What image are you interested in?",
                choices=data_choices,
                validate=non_empty_validation,
                ),
            inquirer.Checkbox(
                'method',
                message="What method are you interested in?",
                choices=method_choices,
                validate=non_empty_validation,
                ),
            inquirer.Confirm(
                'DL_verb',
                message="Should I be verbose for dico. learning methods?",
                default=False,
                ignore=DL_ignore),
        ]
        answers = inquirer.prompt(questions)

        # Get index in choices lists
        data_ind = [data_choices.index(anw) for anw in answers['data']]
        method_ind = [method_choices.index(anw) for anw in answers['method']]

        # DL verbosity
        DL_verb = False if 'DL_verb' not in answers else answers['DL_verb']

        # Generate results.
        inpaint_data(
            answers['data'], data_ind,
            answers['method'], method_ind, DL_verb)

    finally:

        # Set DATA folder bak to previous conf.
        if DATA_bak is not None:
            inpystem.set_data_path(DATA_bak)
