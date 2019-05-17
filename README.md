# LwfSeg-AeiSeg
#### Extending Pretrained Segmentation Networks with Additional Anatomical Structures

Class-incremental learning for segmentation in a lifelong learning scenario.  
The code here implements the work proposed by Ozdemir et. al. on class-incremental learning for segmentation ([pre-print](https://arxiv.org/abs/1811.04634), [journal](https://doi.org/10.1007/s11548-019-01984-4)).  
Currently, the workflow is being refactored to benefit from TF 1.12 (mainly migration to tensorflow.keras). Please report any issues/bugs.  


## Prerequisites

- Python (>=3.3), NumPy (1.15.x), h5py, scikit-image
- Tensorflow(>=1.12.0)

## Data I/O

- Script expects data input in forms of HDF5 files. In order to reproduce experiments with the least amount of overhead, one needs to download SKI10 dataset and generate HDF5 container with resized volumes for training. We have also applied bias field correction using N4ITK.

## Experiment Configurations

More to come!

## Citation

If you use this repository, please consider referring to the following paper:

### LwfSeg/AeiSeg

```
@article{ozdemir2018extending,
  title={Extending Pretrained Segmentation Networks with Additional Anatomical Structures},
  author={Ozdemir, Firat and Goksel, Orcun},
  journal={arXiv preprint arXiv:1811.04634},
  year={2018}
}
```
### CoRiSeg

```
@InProceedings{ozdemir2018learn,
author="Ozdemir, Firat
and Fuernstahl, Philipp
and Goksel, Orcun",
title="Learn the New, Keep the Old: Extending Pretrained Models with New Anatomy and Images",
booktitle="Medical Image Computing and Computer Assisted Intervention (MICCAI)",
year="2018",
publisher="Springer International Publishing",
pages="361--369",
isbn="978-3-030-00937-3"
}
```
The code is developed by Firat Ozdemir `ozdemirf@vision.ee.ethz.ch`, [Computer-assisted Applications in Medicine](http://www.caim.ee.ethz.ch/), Computer Vision Laboratory, ETH Zurich.
