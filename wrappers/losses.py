#Author: Firat Ozdemir, 2018, fozdemir@gmail.com, ETH Zurich


import tensorflow as tf
import numpy as np



def weighted_cross_entropy_loss(logits, labels, class_weights):
    '''Function expects logits and labels in a "categorical" form (one-hot encoded, also supports if classes are
    not mutually exclusive)'''
    n_class = len(class_weights)

    flat_logits = tf.reshape(logits, [-1, n_class])
    flat_labels = tf.reshape(labels, [-1, n_class])

    loss_voxels = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits, labels=flat_labels)

    if not np.all(class_weights == class_weights[0]):
        class_weights_t = tf.constant(np.array(class_weights, dtype=np.float32))
        w_t = tf.math.reduce_sum(tf.math.multiply(flat_labels, class_weights_t), axis=-1) #weight for every voxel.
        loss_voxels = tf.math.multiply(loss_voxels, w_t)
    else:
        print('All classes have same weight, non-weighted loss is going to be applied.')

    scalar_loss = tf.math.reduce_mean(loss_voxels)
    return scalar_loss


def weighted_soft_dice_class_array(logits, labels, class_weights, epsilon=1.e-8):
    '''Function expects logits and labels in a "categorical" form (one-hot encoded, also supports if classes are
        not mutually exclusive)'''
    n_class = len(class_weights)

    flat_logits = tf.reshape(logits, [-1, n_class])
    flat_logits = tf.nn.softmax(flat_logits, axis=1)
    flat_labels = tf.cast(tf.reshape(labels, [-1, n_class]), dtype=tf.float32)

    intersection = tf.math.reduce_sum(tf.math.multiply(flat_logits, flat_labels), axis=0) # |X \cap Y| per class
    x = tf.math.reduce_sum(flat_logits, axis=0) #|X|
    y = tf.math.reduce_sum(flat_labels, axis=0) #|Y|

    dice_array = 2 * intersection / (x + y + epsilon)
    return dice_array

def weighted_soft_dice_score(logits, labels, class_weights, channels_to_ignore=[0], epsilon=1.e-8):

    n_class = len(class_weights)
    dice_array = weighted_soft_dice_class_array(logits=logits, labels=labels, class_weights=class_weights,
                                                epsilon=epsilon)
    channels_of_interest = [i for i in range(n_class) if i not in channels_to_ignore]
    if len(channels_of_interest) == 1:
        return tf.math.reduce_mean(dice_array[...,channels_of_interest[0]])
    elif len(channels_of_interest) > 1:
        fg_array = tf.concat([dice_array[...,i] for i in channels_of_interest], axis=1)
        return tf.math.reduce_mean(fg_array)
    else:
        raise AssertionError('No FG channel!')

