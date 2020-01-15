## DeepEI

This is the repository of codes for the paper entitled "Predicting molecular fingerprint from electron−ionization mass spectrum with deep neural networks" (submit). This repository only contains the source codes without any data or pretrained models, due to the models were trained by NIST dataset.

DeepEI contain two main parts: 1. Predicting molecular fingerprint from EI-MS (*Fingerprint* folder); 2. Predicting retention index from structure (*retention* folder). Each folder contains the codes for data pretreatment, model training and model selection.

Moreover, the *scripts* folder contains the scripts for convert NIST database to numpy object; the *Discussion* folder contains the scripts for evaluating the identification performance, and comparing with [NEIMS](https://github.com/brain-research/deep-molecular-massspec) package. The corresponding results are also included.

**Contact:** ji.hongchao@foxmail.com