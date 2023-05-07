
## Introduction

This is an implementation of submitted paper: 

Desai, S. V., Balasubramanian, V. N., Fukatsu, T., Ninomiya, S., & Guo, W. (2019). Automatic estimation of heading date of paddy rice using deep learning. Plant Methods, 15(1), 76. https://doi.org/10.1186/s13007-019-0457-1 

To plan the perfect time for harvest of rice crops, we detect and quantify the flowering of paddy rice. The dataset of rice crop images used in this work is taken from [1].

## Methodology 

In this paper, we proposed an automatic pipeline for branch–leaf segmentation and leaf phenotypic parameter measurement for pear trees based on lidar point cloud. The method segments branch–leaf point clouds based on the PointNet++ model, extracts single leaf data by mean shift clustering algorithm, and estimates leaf inclination angle, length, width, and area by plane fitting, midrib fitting, and triangulation. It achieved high accuracy in branch–leaf segmentation, single leaf extraction, and leaf phenotypic parameter estimation. 


## Branch-leaf segmentation : 

train 

```bash

python train_partseg.py


```
test
```bash


python test_partseg.py

```

## Single leaf segmentation : 

```bash
python MS_cluster.py
```

## Midrib fitting algorithm

```bash
python test_partseg.py

```



## References
[1] Qi, C.R.; Yi, L.; Su, H.; Guibas, L.J. PointNet++: Deep hierarchical feature learning on point sets in a metric space. In Proceedings of the Proceedings of the 31st International Conference on Neural Information Processing Systems, Long Beach, CA, USA, 4–9 December 2017; pp. 5105–5114.


