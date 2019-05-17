########################
# Author:
# Firat Ozdemir (fozdemir@gmail.com), Copyright (R) 2019, ETH Zurich
########################
# Module contains scripts to reduce tensorboard summary clutter from training pipeline

import tensorflow as tf

class Summary:
    '''Simple class to manage summaries (scalar, image, histogram).
    If a category is passed with the item to be added to summary, all summaries under the corresponding category
    can be merged later on. Note that keys are expected to be unique across different summaries.'''
    def __init__(self, **kwargs):
        self.nlabels = kwargs.get('nlabels', None)
        self.scalars = {}
        self.images = {}
        self.histograms = {}
        self.categories = {}

    def _add_to_category(self, key, category):
        if category is not None:
            if category in self.categories:
                self.categories[category] += [key]
            else:
                self.categories[category] = [key]
    def add_scalar(self, val, key, category=None):
        self._add_to_category(key=key, category=category)
        self.scalars[key] = val
    def add_histogram(self, val, key, category=None):
        self._add_to_category(key=key, category=category)
        self.histograms[key] = val
    def add_intensity_map(self, val, key, category=None, **kwargs):
        self._add_to_category(key=key, category=category)
        shape = kwargs.get('shape', val.get_shape().as_list())
        if len(shape) < 2:
            raise AssertionError('image should have 2+ dims')
        elif len(shape) > 2:
            #see if any dimension can be squeezed
            val = tf.squeeze(val)
            if len(val.get_shape().as_list()) != 2:
                raise AssertionError('image tensor has more than 2 valid dimensions')

        val = val - tf.reduce_min(val)
        val = val / tf.reduce_max(val)
        val = tf.cast(val*255, dtype=tf.uint8)
        val = tf.expand_dims(tf.expand_dims(val, axis=-1), axis=0)
        self.images[key] = val
    def add_label_map(self, val, key, category=None, **kwargs):
        self._add_to_category(key=key, category=category)
        nlabels = kwargs.get('nlabels', self.nlabels)
        if nlabels is None:
            raise AssertionError('Please pass "nlabels" in kwargs.')
        val = val / (nlabels-1)
        val = tf.cast(val*255, dtype=tf.uint8)
        val = tf.expand_dims(tf.expand_dims(val, axis=-1), axis=0)
        self.images[key] = val

    def compile_all(self):
        summaries = [tf.summary.scalar(key, self.scalars[key]) for key in self.scalars]
        summaries += [tf.summary.histogram(key, self.histograms[key]) for key in self.histograms]
        summaries += [tf.summary.image(key, self.images[key]) for key in self.images]
        return tf.summary.merge(summaries)
    def compile_category(self, category):
        '''category: can be str or a list of str'''
        if not isinstance(category, list):
            category = [category]
        summaries = []
        for cat in category:
            if cat not in self.categories:
                raise AssertionError('Unknown category: %s' % (str(cat)))
            l_cat = self.categories[cat]
            summaries += [tf.summary.scalar(key, self.scalars[key]) for key in self.scalars if key in l_cat]
            summaries += [tf.summary.histogram(key, self.histograms[key]) for key in self.histograms if key in l_cat]
            summaries += [tf.summary.image(key, self.images[key]) for key in self.images if key in l_cat]
        return tf.summary.merge(summaries)
