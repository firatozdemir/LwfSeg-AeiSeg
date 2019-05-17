__author__ = 'ozdemirf'
########################
# Author:
# Firat Ozdemir (fozdemir@gmail.com), 2019, ETH Zurich
##
# Module contains data augmentation scripts and
# data generator to feed datasets
#
#
########################

import numpy as np
from itertools import combinations
import skimage
import logging
import scipy.spatial.transform
import scipy.ndimage
from copy import deepcopy

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def _crop_height_and_width(im, target_height, target_width):
    shp = im.shape  # oversized image shape
    if target_height > shp[0] or target_width > shp[1]:
        raise AssertionError('Image smaller than crop region. input shape:%s. Target shape: [%d, %d]' % (
        str(shp), target_height, target_width))
    offset_height = np.floor((shp[0] - target_height) / 2.).astype(np.int32)
    offset_width = np.floor((shp[1] - target_width) / 2.).astype(np.int32)
    return im[offset_height:offset_height + target_height, offset_width:offset_width + target_width, ...]


def _pad_height_and_width(im, target_height, target_width):
    shp = im.shape  # undersized image shape
    if target_height < shp[0] or target_width < shp[1]:
        raise AssertionError('Image larger than padded region. input shape:%s. Target shape: [%d, %d]' % (
        str(shp), target_height, target_width))
    offset_height = np.floor((target_height - shp[0]) / 2.).astype(np.int32)
    offset_width = np.floor((target_width - shp[1]) / 2.).astype(np.int32)
    new_shp = list(shp)
    new_shp[0:2] = [target_height, target_width]
    imreturn = np.zeros((new_shp), dtype=im.dtype)
    imreturn[offset_height:offset_height + shp[0], offset_width:offset_width + shp[1], ...] = im #pad
    return imreturn


def _pad_height_crop_width(im, target_height, target_width):
    shp = im.shape  # input image shape (undersized height, oversized width)
    if target_height < shp[0] or target_width > shp[1]:
        raise AssertionError('Image size unsuitable for padding/cropping. input shape:%s. Target shape: [%d, %d]' % (
        str(shp), target_height, target_width))
    offset_height = np.floor((target_height - shp[0]) / 2.).astype(np.int32)
    offset_width = np.floor((shp[1] - target_width) / 2.).astype(np.int32)
    new_shp = list(shp)
    new_shp[0] = target_height
    imreturn = np.zeros((new_shp), dtype=im.dtype)
    imreturn[offset_height:offset_height + shp[0], ...] = im  # pad
    imreturn = imreturn[:, offset_width:offset_width + target_width, ...]  # crop
    return imreturn


def _crop_height_pad_width(im, target_height, target_width):
    shp = im.shape  # input image shape (oversized height, undersized width)
    if target_height > shp[0] or target_width < shp[1]:
        raise AssertionError('Image size unsuitable for padding/cropping. input shape:%s. Target shape: [%d, %d]' % (
        str(shp), target_height, target_width))
    offset_height = np.floor((shp[0] - target_height) / 2.).astype(np.int32)
    offset_width = np.floor((target_width - shp[1]) / 2.).astype(np.int32)
    new_shp = list(shp)
    new_shp[1] = target_width
    imreturn = np.zeros((new_shp), dtype=im.dtype)
    imreturn[:, offset_width:offset_width + shp[1], ...] = im # pad
    imreturn = imreturn[offset_height:offset_height + target_height, ...]  # crop

    return imreturn


def _flip_up_down(im, **kwargs):
    '''Flip image/volume on height axis with given probability.'''
    channel_dim_exist = kwargs.get('channel_dim_exist', True)
    flip_chance = kwargs.get('flip_chance', 0.5)
    seed = kwargs.get('seed', np.random.randint(0, 1e5))
    prng = np.random.RandomState(seed)
    flip_coin = prng.rand(1)[0] < flip_chance
    if not flip_coin:
        return im
    shp = im.shape
    if channel_dim_exist:  # Expected shapes: [height, width, channels], [depth, height, width, channels]
        if len(shp) == 3:  # image
            im = im[::-1, ...]
        elif len(shp) == 4:  # volume
            im = im[:, ::-1, ...]
        else:
            raise AssertionError('Unexpected input shape: %s' % (str(shp)))
    else:  # Expected shapes: [height, width], [depth, height, width]
        if len(shp) == 2:  # image
            im = im[::-1, ...]
        elif len(shp) == 3:  # volume
            im = im[:, ::-1, ...]
        else:
            raise AssertionError('Unexpected input shape: %s' % (str(shp)))
    return im


def _flip_left_right(im, **kwargs):
    '''Flip image/volume on width axis with given probability.'''
    channel_dim_exist = kwargs.get('channel_dim_exist', True)
    flip_chance = kwargs.get('flip_chance', 0.5)
    seed = kwargs.get('seed', np.random.randint(0, 1e5))
    prng = np.random.RandomState(seed)
    flip_coin = prng.rand(1)[0] < flip_chance
    if not flip_coin:
        return im
    shp = im.shape
    if channel_dim_exist:  # Expected shapes: [height, width, channels], [depth, height, width, channels]
        if len(shp) == 3:  # image
            im = im[:, ::-1, :]
        elif len(shp) == 4:  # volume
            im = im[:, :, ::-1, :]
        else:
            raise AssertionError('Unexpected input shape: %s' % (str(shp)))
    else:  # Expected shapes: [height, width], [depth, height, width]
        if len(shp) == 2:  # image
            im = im[:, ::-1]
        elif len(shp) == 3:  # volume
            im = im[:, :, ::-1]
        else:
            raise AssertionError('Unexpected input shape: %s' % (str(shp)))
    return im


def _flip_front_back(im, **kwargs):  # only for volumes
    '''Flip volume on depth axis (axis=0) with given probability.'''
    channel_dim_exist = kwargs.get('channel_dim_exist', True)
    flip_chance = kwargs.get('flip_chance', 0.5)
    seed = kwargs.get('seed', np.random.randint(0, 1e5))
    prng = np.random.RandomState(seed)
    flip_coin = prng.rand(1)[0] < flip_chance
    if not flip_coin:
        return im
    shp = im.shape
    if channel_dim_exist:  # Expected shapes: [height, width, channels], [depth, height, width, channels]
        if len(shp) == 3:  # image
            raise ValueError('function expects volumes, image passed')
        elif len(shp) == 4:  # volume
            im = im[::-1, ...]
        else:
            raise AssertionError('Unexpected input shape: %s' % (str(shp)))
    else:  # Expected shapes: [height, width], [depth, height, width]
        if len(shp) == 2:  # image
            raise ValueError('function expects volumes, image passed')
        elif len(shp) == 3:  # volume
            im = im[::-1, ...]
        else:
            raise AssertionError('Unexpected input shape: %s' % (str(shp)))
    return im


def _rotate(im, **kwargs):
    '''Rotate images.'''
    channel_dim_exist = kwargs.get('channel_dim_exist', True)
    max_angle = float(kwargs.get('max_angle', 30.))
    unidirectional = kwargs.get('unidirectional', False)  # if false, rotation is applied both directions
    interpolation = kwargs.get('interpolation', 1) #0:NN, 1:bilinear, 3:bicubic
    seed = kwargs.get('seed', np.random.randint(0, 1e5))
    prng = np.random.RandomState(seed)
    shp = im.shape
    tmp_num = prng.uniform(size=1)[0]
    if unidirectional:
        angle = tmp_num * max_angle  # in range [0, max_angle)
    else:
        angle = (tmp_num - 0.5) * 2. * max_angle  # in range [-max_angle, max_angle)

    def _rotate_call_2D(im, angle, interpolation):
        '''interpolation: [0:nearest neighbor, 1:bilinear, 3:bicubic]'''
        if interpolation == 0:  # nearest neighbor (suitable for annotations)
            return skimage.transform.rotate(im, angle=angle, order=0, preserve_range=True, mode='constant')
        elif interpolation == 1 or interpolation == 3:  # bilinear or bicubic
            return skimage.transform.rotate(im, angle=angle, order=interpolation, preserve_range=False, mode='constant')
        else:
            raise AssertionError('Unexpected interpolation value: %s' % (str(interpolation)))
    # rot_mtx = cv2.getRotationMatrix2D((shp[1] / 2., shp[0] / 2.), angle, 1)  # (center, angle, scale_factor)
    if channel_dim_exist:  # Expected shapes: [height, width, channels], [depth, height, width, channels]
        if len(shp) == 3:  # image
            for c in range(shp[-1]):
                im[..., c] = _rotate_call_2D(im[..., c], angle=angle, interpolation=interpolation)
                #                 im[...,c] = skimage.transform.rotate(im[...,c], angle=angle, order=interpolation)
                # im[..., c] = cv2.warpAffine(im[..., c], rot_mtx, (shp[1], shp[0]), flags=interpolation)
        elif len(shp) == 4:  # volume
            raise AssertionError('Volume rotation not yet implemented')
        else:
            raise AssertionError('Unexpected input shape: %s' % (str(shp)))
    else:  # Expected shapes: [height, width], [depth, height, width]
        if len(shp) == 2:  # image
            im = _rotate_call_2D(im, angle=angle, interpolation=interpolation)
            # im = cv2.warpAffine(im, rot_mtx, (shp[1], shp[0]), flags=interpolation)
        elif len(shp) == 3:  # volume
            raise AssertionError('Volume rotation not yet implemented')
        else:
            raise AssertionError('Unexpected input shape: %s' % (str(shp)))
    return im


def _resize(im, **kwargs):
    '''Function resizes the image for the purpose of augmentation. Input image shape is kept via crop or zero pad'''
    channel_dim_exist = kwargs.get('channel_dim_exist', True)
    max_ratio = float(kwargs.get('max_ratio', 1.5))
    only_zoom = kwargs.get('only_zoom', False)
    resize_uniform = kwargs.get('resize_uniform', True)  # if false, each dim is randomly picked.
    interpolation = kwargs.get('interpolation', 1)  # 0:NN, 1:bilinear, 3:bicubic
    random_gauss_blur = kwargs.get('random_gauss_blur', False)  # Defines if anti-aliasing filter is picked randomly or not
    seed = kwargs.get('seed', np.random.randint(0, 1e5))
    prng = np.random.RandomState(seed)
    shp = im.shape
    #     print('channel_dim_exist:%s' % (str(channel_dim_exist)))
    if random_gauss_blur:
        blur_range = (0.5, 2.0)
        blur_sigma_tmp = prng.uniform(low=blur_range[0], high=blur_range[1],size=1)[0]
        blur_sigma = [blur_sigma_tmp, blur_sigma_tmp]
    else:
        blur_sigma = None
    if resize_uniform:
        ratio_tmp = prng.uniform(size=1)[0]
    else:
        if channel_dim_exist:  # Expected shapes: [height, width, channels], [depth, height, width, channels]
            ratio_tmp = prng.uniform(size=len(shp) - 1)
        else:  # Expected shapes: [height, width], [depth, height, width]
            ratio_tmp = prng.uniform(size=len(shp))
    if only_zoom:
        ratios = ratio_tmp * (max_ratio - 1.) + 1.  # range [1, max_ratio)
    else:
        ratios = ratio_tmp * (max_ratio - 1. / max_ratio) + 1. / max_ratio  # range [1/max_ratio, max_ratio)
    if resize_uniform:
        height_ratio = ratios
        width_ratio = ratios
    else:
        height_ratio = ratios[0]
        width_ratio = ratios[1]
    cond1 = height_ratio >= 1.0 and width_ratio >= 1.0
    cond2 = height_ratio < 1.0 and width_ratio < 1.0
    cond3 = height_ratio < 1.0 and width_ratio >= 1.0
    cond4 = height_ratio >= 1.0 and width_ratio < 1.0

    def _post_resize_fix(im, shp):
        d = {'target_height': shp[0], 'target_width': shp[1]}
        if cond1:
            return _crop_height_and_width(im, **d)
        elif cond2:
            return _pad_height_and_width(im, **d)
        elif cond3:
            return _pad_height_crop_width(im, **d)
        elif cond4:
            return _crop_height_pad_width(im, **d)
        else:
            raise AssertionError('Unexpected condition for _post_resize_fix')

    def _resize_call_2D(im, output_shape, interpolation, blur_sigma):
        '''interpolation: [0:nearest neighbor, 1:bilinear, 3:bicubic]'''
        if interpolation == 0:  # nearest neighbor (suitable for annotations)
            return skimage.transform.resize(im, output_shape=output_shape, order=0, preserve_range=True,
                                            anti_aliasing=False, mode='constant')
        elif interpolation == 1 or interpolation == 3:  # bilinear or bicubic
            if blur_sigma is not None:
                # if np.all([output_shape[i] >= im.shape[i] for i in range(len(blur_sigma))]):
                #     blur_sigma = None
                for i in range(len(blur_sigma)):
                    if output_shape[i] >= im.shape[i]:
                        blur_sigma[i] = 0
            return skimage.transform.resize(im, output_shape=output_shape, order=interpolation, preserve_range=True,
                                            anti_aliasing=True, anti_aliasing_sigma=blur_sigma, mode='constant')
        else:
            raise AssertionError('Unexpected interpolation value: %s' % (str(interpolation)))

    if channel_dim_exist:  # Expected shapes: [height, width, channels], [depth, height, width, channels]
        if len(shp) == 3:  # image
            out_shp = np.round([shp[0] * height_ratio, shp[1] * width_ratio]).astype(np.int32)
            for c in range(shp[-1]):
                im_tmp = _resize_call_2D(im[..., c], output_shape=out_shp, interpolation=interpolation, blur_sigma=blur_sigma)
                # im_tmp = skimage.transform.resize(im[..., c], output_shape=out_shp, order=interpolation)
                #                 im_tmp = cv2.resize(im[...,c], dsize=(0,0), fx=shp[0], fy=shp[1], interpolation=interpolation)
                im[..., c] = _post_resize_fix(im_tmp, shp)
        elif len(shp) == 4:  # volume
            raise AssertionError('Volume resize not yet implemented')
        else:
            raise AssertionError('Unexpected input shape: %s' % (str(shp)))
    else:  # Expected shapes: [height, width], [depth, height, width]
        if len(shp) == 2:  # image
            out_shp = np.round([shp[0] * height_ratio, shp[1] * width_ratio]).astype(np.int32)
            im_tmp = _resize_call_2D(im, output_shape=out_shp, interpolation=interpolation, blur_sigma=blur_sigma)
            # im_tmp = skimage.transform.resize(im, output_shape=out_shp, order=interpolation)
            #             im_tmp = cv2.resize(im, dsize=(0,0), fx=shp[0], fy=shp[1], interpolation=interpolation)
            im = _post_resize_fix(im_tmp, shp)
        elif len(shp) == 3:  # volume
            raise AssertionError('Volume resize not yet implemented')
        else:
            raise AssertionError('Unexpected input shape: %s' % (str(shp)))
    return im


class DataGenerator:
    def __init__(self, data, **kwargs):
        '''data_augment_config is a dictionary,
        each key (key should have same name as the augmentation function)
        being another dictionary, containing items for function parameters.
        key specific args can be passed through 'key_specific_params'. This will cause generator to append
        these params to every augmentation function (e.g. NN interpolation for discrete GT annotations)
        Generator pregenerates all data augmentation combinations and samples an augmentation from this set.
        All augmentation options have a uniform prob of being applied.'''
        self.data = data
        self.keys = list(data.keys())
        self.output_shape = kwargs.get('output_shape', None)  # if none, original data shape is respected. Beware, this can cause problems for datasets with inconsistent digital resolutions. If not None, pass a dictionary with function to resize to right dimension
        self.data_augment_config = kwargs.get('data_augment_config', None)  # if None, no augmentation
        self.keys_to_augment = kwargs.get('keys_to_augment',[])  # dictionary keys to which data augmentation should be applied
        self.key_specific_params = kwargs.get('key_specific_params',{})  # e.g. {'y': {'interpolation':0}}. One can also add one_hot encoding as post_augmentation step here.
        self.key_specific_post_processing = kwargs.get('key_specific_post_processing', None)  # e.g. {'masks': {_adjust_label_space:params}}. One can also add one_hot encoding as post_augmentation step here.
        self.shuffle = kwargs.get('shuffle', True)  # shuffle dataset
        self.prng = kwargs.get('prng',np.random.RandomState(seed=91))  # seed for generator prng. Important for reproducibility
        self.augmenting = False
        self.name = kwargs.get('name', None)
        self._DEBUG = True
        if self.data_augment_config is not None:
            self.augmenting = True
            # create a permutation of all augmentation options to decide which augmentation(s) to apply with a single coin flip
            l = []
            augment_fn = list(self.data_augment_config.keys())
            for i in range(len(augment_fn) + 1):
                l = l + list(combinations(augment_fn, i))
            self.augment_function_combinations = l

    def __iter__(self):
        len_data = len(self.data[self.keys[0]])  # assumes all keys have same length
        inds_data = np.arange(len_data)
        if self.shuffle:
            self.prng.shuffle(inds_data)
        for i in range(len_data):
            d = dict()
            for k in self.keys:
                d[k] = self.data[k][inds_data[i], ...]
            if self.augmenting:
                augment_choice = list(self.augment_function_combinations[self.prng.randint(0, len(self.augment_function_combinations))])
                if len(augment_choice) > 0:
                    self.prng.shuffle(augment_choice)  # shuffle the order of random augmentations
                    for aug_fn in augment_choice:
                        aug_fn_args = deepcopy(self.data_augment_config[aug_fn])
                        seed_tmp = self.prng.randint(0, 1e5)
                        aug_fn_args['seed'] = seed_tmp
                        for k in self.keys_to_augment:
                            # logging.info('\n\nName:%s\nk:%s\nself.keys_to_augment:%s\nself.key_specific_params:%s\n:aug_fn_args:%s\n\n\n' %
                            #              (str(self.name), str(k), str(self.keys_to_augment), str(self.key_specific_params), str(aug_fn_args)))
                            if k in self.key_specific_params:
                                d_append = self.key_specific_params[k]
                                for k_tmp in d_append:
                                    aug_fn_args[k_tmp] = d_append[k_tmp]
                                d[k] = aug_fn(d[k], **aug_fn_args)
                            else:
                                d[k] = aug_fn(d[k], **aug_fn_args)
            if self.output_shape is not None:
                for category in self.keys:
                    d[category] = self._resize_output_to_shape(d[category], category)
            if self.key_specific_post_processing is not None:
                for category in self.keys:
                    if category in self.key_specific_post_processing:
                        for fn_post_process in self.key_specific_post_processing[category].keys():
                            fn_args = self.key_specific_post_processing[category][fn_post_process]
                            d[category] = fn_post_process(d[category], **fn_args)

            yield d

    def _resize_output_to_shape(self, im, category):
        output_shape = self.output_shape
        interpolation = 1
        if category in self.key_specific_params:
            if 'interpolation' in self.key_specific_params[category]:
                interpolation = self.key_specific_params[category]['interpolation']
        if interpolation == 0:  # nearest neighbor (suitable for annotations)
            return skimage.transform.resize(im, output_shape=output_shape, order=0, preserve_range=True,
                                            anti_aliasing=False, mode='constant')
        elif interpolation == 1 or interpolation == 3:  # bilinear or bicubic
            return skimage.transform.resize(im, output_shape=output_shape, order=interpolation, preserve_range=True,
                                            anti_aliasing=True, mode='constant')
        else:
            raise AssertionError('Unexpected value for interpolation in kwargs: ', interpolation)

