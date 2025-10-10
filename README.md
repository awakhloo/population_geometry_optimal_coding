# neural population geometry and optimal coding

Code associated with the manuscript *Neural population geometry and optimal coding of tasks with shared latent structure* [1]. 

## reproducing the results

In a python 3.11 environment, install the necessary packages via

```python
pip3 install -r requirements_base.txt
```

Note that the pytorch install is set to a CUDA version, so you may need to modify the relevant line to suit your system. 

The MLP analyses can be reproduced by running `code/run_mlp_exps.py`. Note that you can choose which of the MLP configurations you'd like to use (main text vs. SM) by commenting/uncommenting the relevant lines at the top of the script. The macaque analyses can be reproduced by running  `run-monkey-analysis.py`. Remember to modify the output directory before running these scripts. The simulations and figures presented in Fig. 7 can be reproduced using the `figures/Fig-optimal-embedding.ipynb`notebook. 

The rat analyses in Fig. 8 can be reproduced by first processing the behavioral and electrophysiology data with the `process_rats.py` script (note that you will need to download the dataset, which can be found [here](https://dandiarchive.org/dandiset/000978?search=learning&page=2&sortOption=0&sortDir=-1&showDrafts=true&showEmpty=false&pos=12) [2]). The GLMs can then be fit using `fit-GLMs-rat.py`, and the geometry based analysis can be run using `run-pfc-hp-analysis.py`. (Note we include the results for the best performing GLM models. See below.) For each of these scripts, the `area` variable controls whether the analysis is run on the PFC or CA1 data. 

### DeepLabCut dependencies 

The results in Fig. 6 can be reproduced in the notebook `figures/Fig-deeplabcut.ipynb`. However, by default this uses saved representations taken from a pretrained DeepLabCut network. Note that the results from the pretrained network are saved in an LFS
object (see below). To reproduce Fig. 6 completely from scratch, including retraining the DeepLabCut network, you must first run the script `code/setup_deeplabcut.py`, which can take a few hours to train. The version of DeepLabCut used in the paper is not compatible with Python 3.11, so the `setup_deeplabcut.py` script **must be executed in a separate Python 3.8 virutal environment**. To install the dependencies for this script in its new Python 3.8 virtual environment, run
```python
pip3 install -r requirements_deeplabcut.txt
```
Next edit the field `outdir` at the top of the script `code/setup_deeplabcut.py` to some working directory where intermediate results will be stored and run the script. Finally, change the field `representation_file_directory` at the top of `figures/Fig-deeplabcut.ipynb` to point to `outdir`.

### Pre-processed results

The results for all analyses are included in the repository. However, the deeplabcut model representations and GLM results are somewhat large. If you would like to use these results without running the analyses yourself, you must install `git-lfs` before cloning this repository (or call `git lfs pull` to manually download if you have already cloned the reposoitory before installing LFS). 

### DOI 

[![DOI](https://zenodo.org/badge/1072445857.svg)](https://doi.org/10.5281/zenodo.17315549)


### references

[1] Wakhloo A, Slatton W, Chung SY, *Neural population geometry and optimal coding of tasks with shared latent structure* (2024)

[2] Shin, Justin; Jadhav, Shantanu (2024) Single Day W-Track Learning (Version 0.240511.0307) [Data set]. DANDI archive. https://doi.org/10.48324/dandi.000978/0.240511.0307



