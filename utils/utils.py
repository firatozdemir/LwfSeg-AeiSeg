__author__ = 'ozdemirf'
########################
# Author:
# Firat Ozdemir (fozdemir@gmail.com), 2018, ETH Zurich
########################

import glob
import os
import numpy as np
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def normalize(x):
    mi = x.flatten().min()
    x -= mi
    ma = x.flatten().max()
    return x/ma

def zero_mean(x):
    for i in range(len(x)):
        x[i] -= (x[i].flatten().mean())
    return x.astype('float32')

def clamp_bottom_topPpercent(x, P=5):
    '''
    Function clamps the top and bottom P percent values of x for the purpose
    of preventing outliers to cause major disruption in actual sample/batch distribution
    :param x: input numpy vector/tensor
    :param P: Percentage to clamp P \in [0,100]
    :return: clamped numpy vector/tensor
    '''
    flat_x = x.flatten()
    s_x = np.sort(flat_x)
    len_x = len(flat_x)
    top_lim = s_x[int(len_x*(100-P)/100)]
    bottom_lim = s_x[int(len_x*P/100)]
    return np.clip(x, a_min=bottom_lim, a_max=top_lim)


def standardize(x, epsilon=1.e-5):
    '''
    Function converts input x into zero mean univariance and returns it.
    :param x: input numpy vector/tensor
    :param epsilon: epsilon to prevent division by zero if std(x) is very small
    :return: zero mean univariance x
    '''
    m = x.flatten().mean()
    s = x.flatten().std()
    return (x-m)/(s+epsilon)


def get_class_weights(gt, verbose=True, ignore_inds=[255]):
    '''
    Function returns the weight of each class inversely proportional to the number of label voxel count.
    Note that if a label does not exist in gt, it's ignored.
    :param gt: ground truth data with integer labels of each class.
    :return: class weights
    '''
    if isinstance(gt, list):
        un_items = []
        for item in range(len(gt)):
            un_items.append(np.unique(gt[item]))
        un_items = [subl for i in un_items for subl in i]
        unique_labels = np.sort(np.unique(un_items))
        inds_of_interest = [i for i in unique_labels if i not in ignore_inds]
        count = np.zeros((len(inds_of_interest)), dtype=np.float)
        for item in range(len(gt)):
            gt_curr = gt[item]
            for i in range(len(inds_of_interest)):
                count[i] += np.sum(gt_curr == inds_of_interest[i])
    elif isinstance(gt, dict):
        un_items = []
        for k in gt.keys():
            un_items.append(np.unique(gt[k]))
        un_items = [subl for i in un_items for subl in i]
        unique_labels = np.sort(np.unique(un_items))
        inds_of_interest = [i for i in unique_labels if i not in ignore_inds]
        count = np.zeros((len(inds_of_interest)), dtype=np.float)
        for k in gt.keys():
            gt_curr = gt[k].value
            for i in range(len(inds_of_interest)):
                count[i] += np.sum(gt_curr == inds_of_interest[i])
    else:
        unique_labels = np.sort(np.unique(gt))
        count = np.asarray([np.sum(gt == i) for i in unique_labels if i not in ignore_inds])
    if np.any(unique_labels != np.round(unique_labels)):
        print('!!!!!\nWARNING\nUnique_labels dtype: ', unique_labels.dtype, '\n!!!!!!!!!')
    class_weights = float(np.max(count)) / count.astype(np.float)
    class_weights /= np.sum(class_weights)

    if verbose:
        print('Unique labels are: '+str(unique_labels)+'\nClass weights are: '+str(class_weights))

    return class_weights



class SeedKeeper:
    """ Simple class to manage seeds for experimental repeatability."""
    def __init__(self, initial_seed=1991):
        self.seed = initial_seed
        self.prng = np.random.RandomState(initial_seed)
    def setSeed(self, seed):
        self.prng.seed(seed=seed)
    def fetch(self):
        return self.prng
    def draw_int(self):
        return self.prng.randint(0,1e5)


def pair_iterator_hdf5_samples(images, batch_size=100):
    '''function iterates through the upper triangle of matrix of size images.shape[0] x images.shape[0]'''
    n_images = images.shape[0]
    if batch_size % 2 != 0.0:
        raise AssertionError('Batch size needs to be an even number')

    current_i = 0
    num_ind = batch_size/2
    it_num = int(np.ceil(float(n_images - current_i) / float(num_ind)))
    curr_i = int(current_i)
    # curr_i = int(np.floor(float(current_i) / float(num_ind)))
    logging.info('There will be %d x %d /2 iterations' % (it_num, it_num+1))
    for i in range(it_num):
        if curr_i + num_ind > n_images: #give some of the samples again to match size.
            curr_i = n_images-num_ind
        i_range = np.arange(curr_i, curr_i + num_ind, dtype=np.int32)
        curr_i += num_ind
        num_elm_i = len(i_range)  # not always equal to num_ind
        curr_j = i_range[0]
        for j in range(i, it_num):
            if curr_j + num_ind > n_images:
                curr_j = n_images - num_ind

            j_range = np.arange(curr_j, curr_j + num_ind, dtype=np.int32)

            num_elm_j = len(j_range)
            curr_j += num_ind

            ims_i = images[i_range,...]
            ims_j = images[j_range,...]

            batch_indices = [i_range[0], i_range[0] + num_elm_i, j_range[0],j_range[0] + num_elm_j]
            # logging.info('batch_indices: %s' % (str(batch_indices)))
            try:
                x = np.stack((ims_i, ims_j), axis=0)
                yield x, batch_indices
            except ValueError:
                raise AssertionError('shapes of ims_i: %s, ims_j: %s' % (str(ims_i.shape), str(ims_j.shape)))


