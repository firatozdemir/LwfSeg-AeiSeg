# LwfSeg-AeiSeg
#### Extending Pretrained Segmentation Networks with Additional Anatomical Structures

Class-incremental learning for segmentation in a lifelong learning scenario.  
The code here implements the work proposed by Ozdemir et. al. on class-incremental learning for segmentation ([pre-print](https://arxiv.org/abs/1811.04634), [journal](https://doi.org/10.1007/s11548-019-01984-4)).  
Currently, the workflow is being refactored to benefit from TF 1.12 (mainly migration to tensorflow.keras). Please feel free to report any issues/bugs.  


## Prerequisites

- Python (>=3.3), NumPy (1.15.x), h5py, scikit-image
- Tensorflow(>=1.12.0)

## Data I/O

- Script expects data input in forms of HDF5 files. In order to reproduce experiments with the least amount of overhead, one needs to download SKI10 dataset and generate HDF5 container with resized volumes for training. We have also applied bias field correction using N4ITK.

## Experiment Configurations

Provided script allows for running: CurSeg, IncSeg, finetune, ReSeg, LwfSeg, AeiSeg. 
All relevant functions for CoRiSeg are also provided, however, not tested.

inc_train.py is the main script to do all training. It accepts arguments to set the method, holdout set, and incremental ratio (IR).  
A few key points:  
IR100: --exp_scenario 1  
IR17: --exp_scenario 3  
IR04: --exp_scenario 4  
IR01: --exp_scenario 2  

HO1: --index_shuffle_seed_ID 1  
HO2: --index_shuffle_seed_ID 2  
HO3: --index_shuffle_seed_ID 3  
HO4: --index_shuffle_seed_ID 4  
HO5: --index_shuffle_seed_ID 5  

A few examples to run the proposed method:
```bash
# IR17, HO1: 
# CurSeg:
python inc_train.py --continue_run --index_shuffle_seed_ID 1 --exp_scenario 3 -l unet2D_BN_wxent_SKI10_resize224_1femur_2tibia_HO1_IR17_IJCARS -c unet2D_SKI10_resize_1femur_2tibia_IJCARS2019 -m incremental0 -v 
# IncSeg:
python inc_train.py --continue_run --index_shuffle_seed_ID 1 --exp_scenario 3 -l unet2D_BN_wxent_SKI10_resize224_1femur_2tibia_HO1_IR17_IJCARS/trainOnIncrementalData -c unet2D_SKI10_resize_1femur_2tibia_IJCARS2019 -m incremental0 --init_train_on_incremental -v 
# finetune:
python inc_train.py --continue_run --index_shuffle_seed_ID 1 --exp_scenario 3 -l unet2D_BN_wxent_SKI10_resize224_1femur_2tibia_HO1_IR17_IJCARS -c unet2D_SKI10_resize_1femur_2tibia_IJCARS2019 -m incremental1 -e finetune -v 
# ReSeg:
python inc_train.py --continue_run --index_shuffle_seed_ID 1 --exp_scenario 3 -l unet2D_BN_wxent_SKI10_resize224_1femur_2tibia_HO1_IR17_IJCARS -c unet2D_SKI10_resize_1femur_2tibia_IJCARS2019 -m incremental1 -e reseg --K_uncertain 1000 --k_rep 100 --compute_exemplar_samples -v 
# LwfSeg:
python inc_train.py --continue_run --index_shuffle_seed_ID 1 --exp_scenario 3 -l unet2D_BN_wxent_SKI10_resize224_1femur_2tibia_HO1_IR17_IJCARS -c unet2D_SKI10_resize_1femur_2tibia_IJCARS2019 -m incremental1 -e lwfseg -v 
# AeiSeg:
python inc_train.py --continue_run --index_shuffle_seed_ID 1 --exp_scenario 3 -l unet2D_BN_wxent_SKI10_resize224_1femur_2tibia_HO1_IR17_IJCARS -c unet2D_SKI10_resize_1femur_2tibia_IJCARS2019 -m incremental1 -e aeiseg --K_uncertain 1000 --k_rep 100 --compute_exemplar_samples -v 
```

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
