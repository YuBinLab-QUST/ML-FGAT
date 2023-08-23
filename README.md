# ML-GGAT
Predicting the multi-label protein subcellular localization based on interpretable graph attention networks and feature generative adversarial networks

###ML-GGAT uses the following dependencies:

Python 3.8 numpy pandas scikit-learn tensorflow keras

###Guiding principles:

**The dataset file contains the Gram-positive bacteria dataset, the human dataset, the virus dataset, the Gram-negative bacteria, the plant dataset,the SARS-CoV-2 dataset.

**Feature extraction： CTDC.py,CTDD.py,CTDT.py is the implementation of CTD. CTrisd.py is the implementation of CT. DDE.py is the implementation of DDE. DPC.py is the implementation of DPC. ebgw1.m, ebgw2.m, ebgw3.m, mainEBGW.m is the implementation of EBGW. PsePSSM.m is the implementation of PsePSSM.

**Differential_Evolution: DE.py is the implementation of Differential Evolution.

**Feature_selection: CCA.m is the implementation of CCA. MDDM.m is the implementation of MDDM. MLSI.m is the implementation of MLSI. MVMD.m is the implementation of MVMD. PCA.py is the implementation of PCA. wMLDAe.m is the implementation of wMLDAe.

**Classifier: BiLSTM.py is the implementation of BiLSTM. CNN.py is the implementation of CNN. GAT.py is the implementation of GAT. GRU.py is the implementation of GRU. InsDIF.m is the implementation of InsDIF. MLKNN.py is the implementation of MLKNN. Multi-head.py is the implementation of Multi-head. Rank-SVM.py is the implementation of Rank-SVM. 
