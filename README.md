
## Introduction

This is an implementation of submitted paper.


## Citation
Li, H.; Wu, G.; Tao, S.; Yin, H.; Qi, K.; Zhang, S.; Guo, W.; Ninomiya, S.; Mu, Y. Automatic Branch–Leaf Segmentation and Leaf Phenotypic Parameter Estimation of Pear Trees Based on Three-Dimensional Point Clouds. Sensors 2023, 23, 4572. https://doi.org/10.3390/s23094572


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
python yemai.py

```



## References
[1] Qi, C.R.; Yi, L.; Su, H.; Guibas, L.J. PointNet++: Deep hierarchical feature learning on point sets in a metric space. In Proceedings of the Proceedings of the 31st International Conference on Neural Information Processing Systems, Long Beach, CA, USA, 4–9 December 2017; pp. 5105–5114.

## Supplement
If readers have any questions about this study code, please contact the authors.
Eamil address: haitao971028@gmail.com,yuemu@njau.edu.cn
