# monier2020fast codes

[Fast reconstruction of atomic-scale STEM-EELS images from sparse sampling.](https://www.sciencedirect.com/science/article/abs/pii/S0304399119302499?via%3Dihub)

These codes aim at reproducing the paper results.

## Requirements

The codes rely on the [pystem](https://github.com/etienne-monier/inpystem) python library (version 0.1.1) that I developped for this purpose.

**The code has been tested on python 3.6 only.**

### Using the requirement file

A requirement file is provided to install a new environment. My advice is to use [pyenv](https://github.com/pyenv/pyenv) coupled with [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) to create a virtual environment. Then, simply setup the environment with

```
pip install -r requirements.txt
``` 

### Manual installation

Another way is to manually install the requirements.

```
pip install scikit-image==0.16.2 hyperspy==1.5.2 inpystem==0.1.1 PyQt5 inquirer blessings
``` 

Warning: The inpystem version is 0.1.1. The scikit and hyperspy versions are set as a conflict was discovered recently discovered (see [here](https://github.com/hyperspy/hyperspy/issues/2402)).


## Usage

To reproduce the results, one should first build the reconstructed data to generate the figures afterwards.

### Build data

To build the data, please execute the `reconstruction.py` program:

```
$ python reconstruction.py
```

The data to process and the reconstruction methods to execute will be asked in the console. **Be aware that this is highly time-consuming, expecially for R1 and for dictionary-learning methods**. I recommand you to first try the R2 and S data for some methods and add other configuration afterwards. All output data are stored in the `reconstruction`folder.

### Generate results

To generate the results, the python interpreter should be interactive. We then recommend you to use `ipython` instead of `python` alone. Just type:

```
$ ipython 

Python 3.6.2 (default, Jul 24 2019, 11:45:48) 
Type 'copyright', 'credits' or 'license' for more information
IPython 7.16.1 -- An enhanced Interactive Python. Type '?' for help.

In [1]: %run display.py
```

Again, the program will ask you which figure you are interested in. Finally, it will propose you to display the results, to save the results or to perform both. The results are saved in the results directory.

## Author and license

These codes were written by [Etienne Monier](http://monier.perso.enseeiht.fr/) and are distributed under the MIT license.