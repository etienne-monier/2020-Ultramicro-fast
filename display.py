#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib

import numpy as np

import imageio

import matplotlib.pyplot as plt
import matplotlib

import inquirer
import blessings

import inpystem
import inpystem.tools.metrics as metrics

import pyxport
import add_patch


import pydata
from basis_analysis import FourierApproxError, DctApproxError, \
    WaveletApproxError


def get_available_methods():
    """Returns a dictionary with available methods for all data.

    Returns
    -------
    dict
        Dictionary with available methods for all data.
    """
    method_choices = ['NN', '3S', 'CLS', 'ITKrMM', 'wKSVD', 'BPFA']
    av_methods = {}

    for data in ['R1', 'R2', 'S', 'Real']:

        methods = []

        for m in method_choices:
            p = pathlib.Path('reconstruction') / (data + '_' + m + '.npz')
            if p.exists():
                methods.append(m)

        av_methods[data] = methods

    return av_methods


def print_figure_name_to_console(t, figure):

    print('[{}] {}'.format(t.yellow + 'o' + t.normal, figure))


def produce_output(figures, output):
    """
    """
    t = blessings.Terminal()

    data_list = ['R1', 'R2', 'S', 'Real']
    data_obj = [pydata.R1(), pydata.R2(), pydata.Synth(), pydata.Real()]
    av_methods = get_available_methods()

    #
    # Figure 2 #########################################################
    #
    if 'Figure 2' in figures:
        print_figure_name_to_console(t, 'Figure 2')

        # Table containing the ratios for the plot.
        # The ratios correspund to the ratio of coefficients to keep.
        rTab = np.arange(0, 0.2, 0.001)

        # The labels of bases to test
        labels = ['fourier', 'dct', 'db3', 'db10', 'db20',
                  'sym3', 'sym10', 'sym20']
        N = len(labels)

        ##
        # Compute the metric and store it for each spim.
        for obj in data_obj[:3]:

            # Loading data
            spim = obj.Y

            # Take even shape
            if spim.shape[0] % 2 != 0:
                spim = spim[:-1, :, :]
            if spim.shape[1] % 2 != 0:
                spim = spim[:, :-1, :]

            # Reconstruction error computation
            res = [0 for cnt in range(N)]

            for cnt, label in enumerate(labels):

                if label == 'fourier':
                    res[cnt] = FourierApproxError(spim, rTab)[0]
                elif label == 'dct':
                    res[cnt] = DctApproxError(spim, rTab)[0]
                else:
                    res[cnt] = WaveletApproxError(spim, rTab, label)[0]

            if output in [0, 2]:

                # Displays results
                f, ax = plt.subplots(1, 1)

                for cnt, label in enumerate(labels):
                    ax.semilogy(rTab, res[cnt], label=label)

                ax.set_xlabel('r'),
                ax.set_ylabel('Rec. error'),
                ax.grid(True),
                ax.legend()
                ax.set_title('Figure 2: Data {}'.format(obj.key))

            if output in [1, 2]:

                p = pathlib.Path('results') / 'Figure_2'
                p.mkdir(parents=True, exist_ok=True)

                # Save results
                dico_rec_error = dict(zip(labels, res))
                pyxport.save_dat({'rTab': rTab, **dico_rec_error},
                                 loc=str(p / '{}.dat'.format(obj.key)))

    #
    # Table 3 ##########################################################
    #
    if 'Table 3' in figures:
        print_figure_name_to_console(t, 'Table 3')

        # Produces the metrics table.
        #
        column_labels = ['SNR', 'aSAD (100x)', 'SSIM', 'Time (s)']

        # Each data key will contain a dict with keys:
        #   row_labels
        #   cellText
        data_tables = {}

        for cnt_d, data in enumerate(data_list):

            if data == 'Real' or len(av_methods[data]) == 0:
                continue

            # Will contain data and row labels for the data
            data_dict = {'labels': [],
                         'cellText': []}

            for cnt_m, m in enumerate(av_methods[data]):

                row = []

                # Add label.
                data_dict['labels'].append(m)

                # Add data
                #

                # Path of the data file.
                p = pathlib.Path('reconstruction') / (data + '_' + m + '.npz')
                # Load data
                data_f = np.load(str(p))
                # Get object
                obj = data_obj[cnt_d]

                # Fill data
                # SNR
                row.append('{:.2f}'.format(
                    metrics.SNR(xhat=data_f['xhat'], xref=obj.X)))
                # aSAD
                row.append('{:.3f}'.format(
                    100*metrics.aSAD(xhat=data_f['xhat'], xref=obj.X)))
                # SSIM
                row.append('{:.3f}'.format(
                    metrics.SSIM(xhat=data_f['xhat'], xref=obj.X)))
                # Time
                row.append('{:.2e}'.format(float(data_f['time'])))

                data_dict['cellText'].append(row)

            # Store table
            data_tables[data] = data_dict

        # Generate Latex table
        #
        if output in [1, 2]:

            p = pathlib.Path('results') / 'Table_3'
            p.mkdir(parents=True, exist_ok=True)

            for data, dic in data_tables.items():

                with open(p / data, 'w') as file:

                    text = r"""\begin{tabular}{ccccc}
\tMethod & SNR & aSAD (100$\times$) & \corr{SSIM} & Time(s)\\
"""
                    for cnt_l, label in enumerate(data_dict['labels']):

                        text += "\t{} & {}\\\\\n".format(
                            label, ' & '.join(data_tables[data]['cellText'][cnt_l]))

                    text += r'\end{tabular}'
                    file.write(text)

        # Display table
        #
        if output in [0, 2]:

            for data, dic in data_tables.items():

                # Produces axis.
                fig, ax = plt.subplots()
                ax.axis('off')
                ax.axis('tight')

                # Create table object.
                tab = matplotlib.table.table(
                    ax=ax,
                    rowLabels=dic['labels'],
                    colLabels=column_labels,
                    cellText=dic['cellText'],
                    loc='center')
                tab.scale(1, 4)

                # ax.add_table(tab)
                ax.set_title('Table 3: Data {}'.format(data))

    #
    # Figure 3 #########################################################
    #
    if 'Figure 3' in figures:
        print_figure_name_to_console(t, 'Figure 3')

        # Main image
        file = str(pathlib.Path('acquisitions') / 'Figure_3' / 'ref.png')

        im = imageio.imread(file)

        zoom_im = add_patch.rect_patch(im, ulpix=(10, 5), lrpix=(30, 25))
        zoom_im = add_patch.pix_patch(zoom_im, pix=(21, 13), c='b')
        # zoom_im = pix_patch(zoom_im, pix=(19, 18), c='r')

        small_im = zoom_im[11:30, 6:25, :]

        # Mask zoom
        file = str(pathlib.Path('acquisitions') / 'Figure_3' / 'mask.png')
        im = imageio.imread(file)

        zoom_im_mask = add_patch.pix_patch(im, pix=(21, 13), c='b')
        small_im_mask = zoom_im_mask[11:30, 6:25, :]

        # Save images
        if output in [1, 2]:
            p = pathlib.Path('results') / 'Figure_3'
            p.mkdir(parents=True, exist_ok=True)

            imageio.imwrite(str(p / 'rectangle.png'), zoom_im)
            imageio.imwrite(str(p / 'zoom_rec.png'), small_im)
            imageio.imwrite(str(p / 'zoom_rec_mask.png'), small_im_mask)

        # Display output
        if output in [1, 2]:

            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(zoom_im)
            ax[0].set_title('Band #2 of R2')

            ax[1].imshow(small_im)
            ax[1].set_title('Zoom')

            ax[2].imshow(small_im_mask)
            ax[2].set_title('Zoom (Mask)')

            fig.suptitle('Figure 3')

            for cnt in range(3):
                ax[cnt].axis('off')

    #
    # Figure 4 #########################################################
    #
    if 'Figure 4' in figures:
        print_figure_name_to_console(t, 'Figure 4')

        if len(av_methods['R2']) == 0:
            print('\t{}No R2 method available. Skipping Figure 4.{}'.format(
               t.red + t.bold, t.normal))

        else:

            # Data object
            obj = data_obj[1]
            # Non-sampled pixel location
            loc = [19, 18]
            loc_s = (np.s_[loc[0]], np.s_[loc[1]], np.s_[:])

            # Structures to save output
            Name = []
            Spectra = []

            # Ev Table
            eV = 494.320 + np.arange(obj.X.shape[-1]) * 0.324 - \
                818.89 + 834.22

            # ground truth
            Name.append('Reference')
            Spectra.append(obj.X[loc_s])

            for m in av_methods['R2']:

                # Get data
                # Path of the data file.
                p = pathlib.Path('reconstruction') / ('R2_' + m + '.npz')
                # Load data
                data_f = np.load(str(p))

                # Store output
                Name.append(m)
                Spectra.append(data_f['xhat'][loc_s])

            # Save spectra
            if output in [1, 2]:

                p = pathlib.Path('results') / 'Figure_4'
                p.mkdir(parents=True, exist_ok=True)

                dico = dict(zip(Name, Spectra))
                dico['eV'] = eV

                pyxport.save_dat(
                    data=dico,
                    loc=str(p / 'figure-4.dat'))

            # Display output
            if output in [1, 2]:

                fig, ax = plt.subplots()

                for cnt in range(len(Name)):
                    ax.plot(eV, Spectra[cnt], label=Name[cnt])

                ax.set_xlabel('Energy loss (eV)')
                ax.set_ylabel('Amplitude')
                ax.set_title('Figure 4')
                ax.legend()

    #
    # Figure 5 #########################################################
    #
    if 'Figure 5' in figures:
        print_figure_name_to_console(t, 'Figure 5')

        if len(av_methods['Real']) == 0:
            print('\t{}No Real method available. Skipping Figure 5.{}'.format(
               t.red + t.bold, t.normal))

        else:

            # Object and methods.
            obj = data_obj[3]
            methods = av_methods['Real']

            # Interest bands.
            bands = [(np.s_[:], np.s_[:], np.s_[70:76]),
                     (np.s_[:], np.s_[:], np.s_[990:996]),
                     (np.s_[:], np.s_[:], np.s_[1448:1454])]

            # Mask
            pix = 0.2
            m, n, B = obj.X.shape
            scan = inpystem.Scan.random((m, n), ratio=pix, seed=0)
            mask = scan.get_mask()

            # Structures
            Labels = []
            Images = []

            # Produce Reference.
            Labels.append('Reference')

            tmp = []
            for c_b, b_slice in enumerate(bands):
                tmp.append(np.sum(obj.X[b_slice], 2))
            Images.append(tmp)

            # For other methods
            for m in methods:

                # Get data
                # Path of the data file.
                p = pathlib.Path('reconstruction') / ('R2_' + m + '.npz')
                # Load data
                data_f = np.load(str(p))

                # Produce images
                tmp = []
                for c_b, b_slice in enumerate(bands):
                    tmp.append(np.sum(data_f['xhat'][b_slice], 2))

                Labels.append(m)
                Images.append(tmp)

            # Save images
            if output in [1, 2]:

                p = pathlib.Path('results') / 'Figure_5'
                p.mkdir(parents=True, exist_ok=True)

                for cnt in range(len(Labels)):
                    for cnt_b in range(len(bands)):

                        pyxport.plot2im(
                            mat=Images[cnt][cnt_b],
                            loc=str(p / '{}_{}.png'.format(
                                Labels[cnt], cnt_b)),
                            cmap='gray')

                # Add mask
                pyxport.plot2im(
                    mat=mask,
                    loc=str(p / 'Mask.png'),
                    cmap='gray')

            # Print output
            if output in [0, 2]:

                fig, ax = plt.subplots(len(Labels)+1, 3)

                for cnt in range(len(Labels)):
                    for cnt_b in range(3):

                        ax[cnt, cnt_b].matshow(Images[cnt][cnt_b])
                        ax[cnt, cnt_b].tick_params(labelsize=0)

                        if cnt_b == 0:
                            ax[cnt, cnt_b].set_ylabel(Labels[cnt])

                ax[len(Labels), 0].matshow(mask)
                ax[len(Labels), 0].set_ylabel('Sampl. mask')

                ax[len(Labels), 0].tick_params(labelsize=0)
                for cnt in range(2):
                    ax[len(Labels), cnt+1].axis('off')

    # Figure 6 #########################################################
    #
    if 'Figure 6' in figures:
        print_figure_name_to_console(t, 'Figure 6')

        if len(av_methods['Real']) == 0:
            print('\t{}No Real method available. Skipping Figure 6.{}'.format(
               t.red + t.bold, t.normal))

        else:

            # Data object
            obj = data_obj[1]
            # Non-sampled pixel location
            loc = [19, 18]
            loc_s = (np.s_[loc[0]], np.s_[loc[1]], np.s_[:])

            # Structures to save output
            Name = []
            Spectra = []

            # ground truth
            Name.append('Reference')
            Spectra.append(obj.X[loc_s])

            # Ev Table
            eV = 494.320 + np.arange(obj.X.shape[-1]) * 0.324 - \
                818.89 + 834.22

            for m in av_methods['Real']:

                # Get data
                # Path of the data file.
                p = pathlib.Path('reconstruction') / ('Real_' + m + '.npz')
                # Load data
                data_f = np.load(str(p))

                # Store output
                Name.append(m)
                Spectra.append(data_f['xhat'][loc_s])

            # Save spectra
            if output in [1, 2]:

                p = pathlib.Path('results') / 'Figure_6'
                p.mkdir(parents=True, exist_ok=True)

                dico = dict(zip(Name, Spectra))
                dico['eV'] = eV

                pyxport.save_dat(
                    data=dico,
                    loc=str(p / 'figure-6.dat'))

            # Display output
            if output in [1, 2]:

                fig, ax = plt.subplots()

                for cnt in range(len(Name)):
                    ax.plot(eV, Spectra[cnt], label=Name[cnt])

                ax.set_xlabel('Energy loss (eV)')
                ax.set_ylabel('Amplitude')
                ax.set_title('Figure 6')
                ax.legend()

    plt.show()


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


if __name__ == '__main__':

    # Options for questions.
    output_choices = [
        'I want to display the results directly.',
        'I want the results to be saved to output directory only.',
        'I want both.']
    figure_choices = ['Figure 2', 'Table 3', 'Figure 3', 'Figure 4',
                      'Figure 5', 'Figure 6']

    # Questions to select the data and methods.
    questions = [
        inquirer.Checkbox(
            'figure',
            message="What figure are you interested in?",
            choices=figure_choices,
            validate=non_empty_validation,
            ),
        inquirer.List(
            'output',
            message="How do you want the results to be generated?",
            choices=output_choices
            ),
    ]
    answers = inquirer.prompt(questions)

    produce_output(
        answers['figure'],
        output_choices.index(answers['output'])
    )
