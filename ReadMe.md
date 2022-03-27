## DeepEI

This is the repository of codes for the paper entitled "Predicting a Molecular Fingerprint from an Electron Ionization Mass Spectrum with Deep Neural Networks" (DOI: [10.1021/acs.analchem.0c01450](https://pubs.acs.org/doi/10.1021/acs.analchem.0c01450)).   

We exported each mass spectrum file (msp file) and molecular file (sdf) from NIST 2017 manually and save to db file (see *scripts/NIST2DB.py* for codes). When training,  we retrieved information from the db file. However, this repository only contains the source codes **without** any data or pretrained models, due to the models were trained by NIST dataset.    

DeepEI contain two main parts: 1. Predicting molecular fingerprint from EI-MS (*Fingerprint* folder); 2. Predicting retention index from structure (*retention* folder). Each folder contains the codes for data pretreatment, model training and model selection.    

Moreover, the *scripts* folder contains the scripts for convert NIST database to numpy object; the *Discussion* folder contains the scripts for evaluating the identification performance, and comparing with [NEIMS](https://github.com/brain-research/deep-molecular-massspec) package. The corresponding results are also included.

**Contact:** ji.hongchao@foxmail.com
