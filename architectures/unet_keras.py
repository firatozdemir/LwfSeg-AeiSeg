# Copyright (C) Firat Ozdemir, ETH Zurich, 2018.
#Author: Firat Ozdemir, ETH Zurich, 2018, fozdemir@gmail.com

import numpy as np
import tensorflow as tf
import tensorflow.keras as krs

class UNet_keras:
    '''Class for Unet with Keras layers (Need TF12.0 or above)'''
    def __init__(self, images, training, nlabels, num_filters_first_layer, padding_type='SAME', architecture_mods='default', **kwargs):
        self.uses_keras_layers = True
        self.dropout_prob = kwargs.get('dropout_rate', 0.0)
        self.l1r = kwargs.get('l1r', 0.0)
        self.l2r = kwargs.get('l2r', 0.0)
        self.use_BN = kwargs.get('use_BN', True) #Use Batch Normalization
        self.architecture_mods = str.lower(architecture_mods)
        self.input_image = images
        self.train_pl = training
        self.classes = nlabels
        self.num_filters_first_layer = num_filters_first_layer
        self.batch_size = tf.shape(images)[0]
        self.padding_type = padding_type
        self.abstraction_layer = None  # will be assigned in self.model()
        self.network_scope = 'unet'
        self.top_scope = tf.get_variable_scope()
        if self.use_BN:
            self.kernel_init = krs.initializers.he_normal()
            self.conv_act = None
            self.use_bias = False
        else:
            self.kernel_init = krs.initializers.glorot_uniform()
            self.conv_act = krs.layers.ReLU()
            self.use_bias = True


        with tf.variable_scope(name_or_scope=self.network_scope):
            self.body_end = self.model_fn()
        self.body_global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.network_scope)
        self.body_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.network_scope)
        self.additional_global_variables = dict()
        self.additional_update_ops = dict()
        if self.architecture_mods == 'default':
            self.logits = self.conv2d_top(input=self.body_end, nf=self.classes, name='logits')

            self.model = tf.keras.Model(inputs=self.krs_inputs, outputs=self.logits)
            # keras BatchNorm moving mean & variance not added in UpdateOps.
            self.body_update_ops += self.model.updates
            print(self.model.summary())



    def softmax_and_predict(self, logits):
        softmax = tf.nn.softmax(logits)
        mask = tf.argmax(softmax, axis=-1)
        return softmax, mask

    def conv2d(self, input, nf, name, kernel_size=(3,3)):
        c = krs.layers.Conv2D(activation=self.conv_act, filters=nf, name=name,
                              kernel_regularizer=krs.regularizers.l1_l2(l1=self.l1r, l2=self.l2r),
                              kernel_size=kernel_size, padding=self.padding_type, use_bias=self.use_bias,
                              kernel_initializer=self.kernel_init, activity_regularizer=None)(input)
        if self.use_BN:
            c1 = krs.layers.BatchNormalization(name=name+'_BN')(c, training=self.train_pl)
            c2 = krs.layers.Activation(krs.layers.ReLU(), name=name+'_act')(c1)
            return c2
        else:
            return c

    def conv2d_top(self, input, nf, name, kernel_size=(1,1), padding='VALID'):
        c = krs.layers.Conv2D(activation=None, filters=nf, name=name,
                              kernel_regularizer=krs.regularizers.l1_l2(l1=self.l1r, l2=self.l2r),
                              kernel_size=kernel_size, padding=padding, use_bias=self.use_bias,
                              kernel_initializer=self.kernel_init, activity_regularizer=None)(input)
        if self.use_BN:
            c1 = krs.layers.BatchNormalization(name=name+'_BN')(c, training=self.train_pl)
            return c1
        else:
            return c

    def conv2dtranspose(self, input, nf, name, kernel_size=(4,4), strides=(2,2)):
        c = krs.layers.Conv2DTranspose(activation=self.conv_act, name=name, padding=self.padding_type, filters=nf,
                                       kernel_size=kernel_size, strides=strides, use_bias=self.use_bias)(input)
        if self.use_BN:
            c1 = krs.layers.BatchNormalization(name=name+'_BN')(c, training=self.train_pl)
            c2 = krs.layers.Activation(krs.layers.ReLU(), name=name+'_act')(c1)
            return c2
        else:
            return c

    def dropout(self, input, name, rate=None):
        if rate is None:
            rate = self.dropout_prob
        return krs.layers.SpatialDropout2D(rate=rate, name=name)(input, training=self.train_pl)

    def new_head(self, numClasses=None, head_name=None):
        nf = self.num_filters_first_layer
        if numClasses is None:
            numClasses = self.classes

        if head_name is None:
            head_name_scope = ''
        else:
            head_name_scope = head_name+'/'

        # with tf.variable_scope(name_or_scope=head_name): #scope does not lead unique tensor names in tf.keras
        conv9_3 = self.conv2d(nf=nf, name=head_name_scope+'conv9_3', input=self.body_end)
        drop_6 = self.dropout(input=conv9_3, name=head_name_scope+'dropout_6')
        conv9_4 = self.conv2d(nf=nf, name=head_name_scope+'conv9_4', input=drop_6)
        drop_7 = self.dropout(input=conv9_4, name=head_name_scope + 'dropout_7')
        pred = self.conv2d_top(input=drop_7, nf=numClasses, name=head_name_scope+'pred')


        self.additional_global_variables[head_name] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                        scope=head_name)
        self.additional_update_ops[head_name] = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=head_name)

        return pred


    def model_fn(self):
        inputs = self.input_image
        padding_type = self.padding_type
        batch_size = self.batch_size
        nf = self.num_filters_first_layer

        if str.lower(padding_type) == 'same':
            input_padded = inputs
        elif str.lower(padding_type) == 'valid':
            num_pad = 92
            additional_padding_needed = (16 * (
                    1 - ((np.asarray(inputs.get_shape().as_list()[1:3]) + num_pad * 2 - 60) / 16 % 1.0)) % 16.) / 2.
            if not np.all(i // 1 == i / 1 for i in additional_padding_needed):
                print('Input image size incompatible!')
            additional_padding_needed = additional_padding_needed.astype(np.int)
            np2 = num_pad + additional_padding_needed
            input_padded = tf.pad(inputs, [[0, 0], [np2[0], np2[0]], [np2[1], np2[1]], [0, 0]], 'CONSTANT')
        else:
            raise AssertionError('Unexpected padding type ', padding_type)

        self.krs_inputs = tf.keras.Input(tensor=input_padded, batch_size=batch_size,
                                         shape=input_padded.get_shape().as_list()[1:], name='input')  # Returns a placeholder tensor


        conv1_1 = self.conv2d(nf=nf, name='conv1_1', input=self.krs_inputs)
        conv1_2 = self.conv2d(nf=nf, name='conv1_2', input=conv1_1)


        pool1 = krs.layers.MaxPool2D(pool_size=(2, 2), padding=padding_type, strides=(2, 2))(conv1_2)
        conv2_1 = self.conv2d(nf=nf*2, name='conv2_1', input=pool1)
        conv2_2 = self.conv2d(nf=nf*2, name='conv2_2', input=conv2_1)


        pool2 = krs.layers.MaxPool2D(pool_size=(2, 2), padding=padding_type, strides=(2, 2))(conv2_2)
        conv3_1 = self.conv2d(nf=nf * (2 ** 2), name='conv3_1', input=pool2)
        conv3_2 = self.conv2d(nf=nf * (2 ** 2), name='conv3_2', input=conv3_1)


        pool3 = krs.layers.MaxPool2D(pool_size=(2, 2), padding=padding_type, strides=(2, 2))(conv3_2)
        conv4_1 = self.conv2d(nf=nf * (2 ** 3), name='conv4_1', input=pool3)
        conv4_2 = self.conv2d(nf=nf * (2 ** 3), name='conv4_2', input=conv4_1)


        pool4 = krs.layers.MaxPool2D(pool_size=(2, 2), padding=padding_type, strides=(2, 2))(conv4_2)
        drop_0 = self.dropout(input=pool4, name='dropout_0')
        conv5_1 = self.conv2d(nf=nf * (2 ** 4), name='conv5_1', input=drop_0)
        conv5_2 = self.conv2d(nf=nf * (2 ** 4), name='conv5_2', input=conv5_1)

        self.abstraction_layer = conv5_2

        drop_1 = self.dropout(input=conv5_2, name='dropout_1')
        upconv4 = self.conv2dtranspose(nf=nf * (2 ** 4), name='upconv4', input=drop_1)
        concat4 = krs.layers.Concatenate(axis=3)([upconv4, conv4_2])
        conv6_1 = self.conv2d(nf=nf * (2 ** 3), name='conv6_1', input=concat4)
        conv6_2 = self.conv2d(nf=nf * (2 ** 3), name='conv6_2', input=conv6_1)


        drop_2 = self.dropout(input=conv6_2, name='dropout_2')
        upconv3 = self.conv2dtranspose(nf=nf * (2 ** 3), name='upconv3', input=drop_2)
        concat3 = krs.layers.Concatenate(axis=3)([upconv3, conv3_2])
        conv7_1 = self.conv2d(nf=nf * (2 ** 2), name='conv7_1', input=concat3)
        conv7_2 = self.conv2d(nf=nf * (2 ** 2), name='conv7_2', input=conv7_1)

        drop_3 = self.dropout(input=conv7_2, name='dropout_3')
        upconv2 = self.conv2dtranspose(nf=nf * (2 ** 2), name='upconv2', input=drop_3)
        concat2 = krs.layers.Concatenate(axis=3)([upconv2, conv2_2])
        conv8_1 = self.conv2d(nf=nf * (2), name='conv8_1', input=concat2)
        conv8_2 = self.conv2d(nf=nf * (2), name='conv8_2', input=conv8_1)

        drop_4 = self.dropout(input=conv8_2, name='dropout_4')
        upconv1 = self.conv2dtranspose(nf=nf * (2), name='upconv1', input=drop_4)
        concat1 = krs.layers.Concatenate(axis=3)([upconv1, conv1_2])
        conv9_1 = self.conv2d(nf=nf, name='conv9_1', input=concat1)
        conv9_2 = self.conv2d(nf=nf, name='conv9_2', input=conv9_1)

        drop_5 = self.dropout(input=conv9_2, name='dropout_5')
        return drop_5