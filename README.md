## Domain Adaptation in Semantic Segmentation
#### Source files submitted in conjunction with M.Eng project report

This repository contains files used to train single-domain and domain adaptation models using the CARLA simulator and Berkeley Deep Drive datasets. 

### Dependencies
The following external libaries are used for implementation:
1) Python v3.6.5
2) TensorFlow GPU v1.8
3) OpenCV v3.4.1
4) scikit-learn v0.19.1
5) ImageIO v2.3.0
6) NumPy v1.14.3
7) Pickle (version unknown, but most should work)

### Configuration
Certain configuration is required before network files can be executed:
1) Datasets are not included due to their size and must be obtained seperately. CARLA images are collected from the [CARLA Simulator](http://www.carla.org). The real-world dataset can be found from [Berkeley Deep Drive](http://bdd-data.berkeley.edu/). They have datasets for multiple preception tasks, only the semantic segmentation dataset is used.
2) Pretrained weights are used for network initialization, courtesy of [Andrea Palazzi](https://github.com/ndrplz/dilation-tensorflow), and can be found [here](https://drive.google.com/open?id=0Bx9YaGcDPu3XR0d4cXVSWmtVdEE).
3) The paths to these weights and data files must be set in the source code. Instructions on this can be found within the code files themselves.
4) For single-domain implementaton, either CARLA or BDD datasets can be used. Depending on which dataset is being trained on, image configuration must be commented/uncommented. Details can be found within the code files themselves. Domain adaptation implementation requires both datasets.

### Included Files
1) helper_functions_bdd.py: helper functions used for loading and processing BDD images.
2) helper_functions_carla.py: helper functions used for loading and processing CARLA images.
3) network_single_domain.ipynb: source code for creating and training network in single-domain implementation.
4) retrain_single_domain.ipynb: source code for retraining saved single-domain models.
5) network_domain_adaptation.ipynb: source code for creating and training network in domain adaptation implementation.
6) retrain_domain_adaptation.ipynb: source code for retraining saved domain adaptation models.
