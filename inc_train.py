########################
# Author:
# Firat Ozdemir (fozdemir@gmail.com), Copyright (R) 2018, ETH Zurich
########################

import os

_USING_SGE_GPU = False
if "CUDA_VISIBLE_DEVICES" in os.environ:
    _USING_SGE_GPU = True
    import matplotlib as mpl
    mpl.use('Agg')

import numpy as np
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
import time
import shutil

import logging
import utils.image
import utils.utils
import external.utils
import h5py
import argparse
import importlib
import architectures.unet_keras
import pre_inc_train
import wrappers.losses
import wrappers.tensorboard as tfboard


def train(log_dir, continue_run, exp_config, verbose=False, _DEBUG=False, additional_save_info=None, **kwargs):
    init_step = 0
    # 'default', 'incremental0', 'incremental1'
    architecture_mods = kwargs.get('architecture_mods', None)
    # Lookup the latest trained model weights (if available)
    initial_log_dir = kwargs.get('initial_log_dir', log_dir)
    init_train_on_incremental = kwargs.get('init_train_on_incremental', False)

    if continue_run:
        try:
            init_checkpoint_path = external.utils.get_latest_model_checkpoint_path(log_dir, 'model.ckpt')
            logging.info('Checkpoint path: %s' % init_checkpoint_path)
            # plus 1 b/c otherwise starts with eval
            init_step = int(init_checkpoint_path.split('/')[-1].split('-')[-1]) + 1
            logging.info('Latest step was: %d' % init_step)
        except:
            continue_run = False
            logging.warning('!!! Did not find init checkpoint. Maybe first run failed. Disabling continue mode...')
    if not continue_run:
        if architecture_mods == 'incremental1':
            try:  # load Head_0 from initial training
                use_best_validation_dice = True
                if use_best_validation_dice:
                    init_checkpoint_path = external.utils.get_latest_model_checkpoint_path(initial_log_dir,
                                                                                      'model_best_dice.ckpt')
                else:
                    init_checkpoint_path = external.utils.get_latest_model_checkpoint_path(initial_log_dir, 'model.ckpt')
                logging.info('Checkpoint path found in initial training step: %s' % init_checkpoint_path)
                # plus 1 b/c otherwise starts with eval
                init_step = 0
                logging.info('Going to start from step: %d' % init_step)
            except:
                raise AssertionError('!!! Did not find init checkpoint. Maybe first run failed. Exiting..')
        else:
            logging.info('Initialize training from scratch.')


    # Create a directory to write source files (might be handy for retrospectively debugging)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    log_dir_source = os.path.join(log_dir, 'source')
    if not os.path.isdir(log_dir_source):
        os.mkdir(log_dir_source)


    if architecture_mods is None or architecture_mods == 'default':
        raise AssertionError('Missing architecture_mods parameter or entered "default"; which is not meant for this pipeline.')
    elif architecture_mods == 'incremental0':
        exp_config_current = exp_config
        modify_GT = exp_config.modify_GT_Head_0
        if init_train_on_incremental: #we need labels of incremental GT
            modify_GT = exp_config.modify_GT_Head_1

    elif architecture_mods == 'incremental1':
        exp_config_current = exp_config.Iteration1
        modify_GT = exp_config.modify_GT_Head_1
        exemplar_info_filename = kwargs.get('exemplar_info_filename', None)
        if exemplar_info_filename is None:
            raise AssertionError('exemplar_info_filename parameter is needed for incremental learning.')
    exp_type = kwargs.get('exp_type', None)
    if exp_type is None:
        raise AssertionError('Missing exp_type parameter.')
    if exp_type in ['aeiseg', 'coriseg', 'reseg']:
        train_ratio_exemplar_vs_new = kwargs.get('train_ratio_exemplar_vs_new', 0.5)
        logging.info('train_ratio_exemplar_vs_new: %.3f' % (train_ratio_exemplar_vs_new))
    elif exp_type in ['finetune', 'lwfseg']:
        train_ratio_exemplar_vs_new = kwargs.get('train_ratio_exemplar_vs_new', -1.0)
        logging.info('train_ratio_exemplar_vs_new: %.3f' % (train_ratio_exemplar_vs_new))

    # Create a RandomState to be passed around for any random generation (repeatability)
    seed_keeper = exp_config.seed_keeper

    # Load dataset
    h5FileName = os.path.join(exp_config_current.dataset_parent_folder, exp_config_current.dataset)
    data = h5py.File(h5FileName, "r")

    inds_train = exp_config_current.inds_train
    inds_validation = exp_config_current.inds_validation
    inds_test = exp_config_current.inds_test
    if verbose:
        print(list(data.keys()))
        logging.info('#items in training: %i, validation: %i, test: %i' % (
            len(inds_train), len(inds_validation), len(inds_test)))



    x_train2D, y_train2D = data['images'][inds_train,...], data['masks'][inds_train, ...]
    x_validation2D, y_validation2D = data['images'][inds_validation,...], data['masks'][inds_validation, ...]

    if verbose:
        patientIDs_val = np.asarray(list(data['PatientID'][inds_validation, ...]))
        uniquePatientIDs_val = np.unique(patientIDs_val)
        patientIDs_train = np.asarray(list(data['PatientID'][inds_train, ...]))
        uniquePatientIDs_train = np.unique(patientIDs_train)
        uniquePatientIDs_test = np.unique(np.asarray(list(data['PatientID'][inds_test, ...])))
        logging.info('Seed: %d, [1991, 1881, 905, 42]\nTraining:Patient IDs are: %s\nValidation: Patient IDs are: %s\nTest: Patient IDs are: %s' %
                     (exp_config.initial_seed, str(uniquePatientIDs_train),str(uniquePatientIDs_val),str(uniquePatientIDs_test)))


    if len(x_train2D.shape) == 3: #single-channel input
        num_input_channels = 1
    elif len(x_train2D.shape) == 4: #multi-channel input
        num_input_channels = x_train2D.shape[3]
    else:
        raise AssertionError(
            'Unexpected number of dimensions in input images: %d' % len(x_train2D.shape))

    try:
        y_train2D = modify_GT(y_train2D)
        y_validation2D = modify_GT(y_validation2D)
        if verbose:
            logging.info('GT contains %s labels.' %
                         (str(np.unique(y_train2D))))
    except:
        print('modify_GT function does not exist.')

    if architecture_mods == 'incremental1':

        if init_train_on_incremental:
            raise AssertionError('init_train_on_incremental is enabled in incremental1 mode.')
        logging.info('Incremental Learning data load: Loading validation and exemplar set from the initial training.')
        h5FileName_head0 = os.path.join(exp_config.dataset_parent_folder, exp_config.dataset)
        data_head0 = h5py.File(h5FileName_head0, "r")
        inds_validation_head0 = exp_config.inds_validation
        y_validation2D_head0 = data_head0['masks'][inds_validation_head0, ...]
        y_validation2D_head0 = exp_config.modify_GT_Head_0(y_validation2D_head0) #Get the validation set labels for head0 training (Head_0)

        nlabels_head0 = exp_config.nlabels
        class_weights_head0 = np.ones((nlabels_head0,))/nlabels_head0 #dummy assignment for finetune

        if exp_type in ['lwfseg', 'aeiseg', 'coriseg', 'reseg']:

            #Load exemplar samples from head0 dataset
            if exp_type in ['aeiseg', 'coriseg', 'reseg']:
                data_exemplar = h5py.File(exemplar_info_filename, "r")
                inds_exemplar = data_exemplar['indices']
                x_train2D_exemplar, y_train2D_exemplar = data_exemplar['images'], data_exemplar['masks']
                sm_logits_exemplar_frozen = data_exemplar['logits']

                if len(inds_exemplar) < exp_config_current.batch_size:
                    logging.info('Too few exemplar data samples, will have to stack them.')
                    x_train2D_exemplar = np.asarray(x_train2D_exemplar)

                    while len(x_train2D_exemplar) < exp_config_current.batch_size:
                        logging.info('Current #exemplar samples: %d, Batch size: %d' % (len(x_train2D_exemplar), exp_config_current.batch_size))
                        x_train2D_exemplar = np.concatenate((np.asarray(x_train2D_exemplar), np.asarray(x_train2D_exemplar)), axis=0)
                        y_train2D_exemplar = np.concatenate((np.asarray(y_train2D_exemplar), np.asarray(y_train2D_exemplar)), axis=0)
                        sm_logits_exemplar_frozen = np.concatenate((np.asarray(sm_logits_exemplar_frozen), np.asarray(sm_logits_exemplar_frozen)), axis=0)



                logging.info('Class labels: exemplar data: %s; new data: %s' % (str(np.unique(y_train2D_exemplar)),
                                                                                str(np.unique(y_train2D))))

                class_weights_head0 = np.asarray(data_exemplar['class_weight'])
            elif exp_type == 'lwfseg':
                if os.path.isfile(exemplar_info_filename):
                    data_exemplar = h5py.File(exemplar_info_filename, "r")
                    class_weights_head0 = np.asarray(data_exemplar['class_weight'])
                else: #cumbersome, but need to load head0 dataset just to compute class_weight
                    logging.info('Class weight priors for old data was not stored, computing again..')
                    y_init_tmp = data_head0['masks'][exp_config.inds_train,...]
                    y_init_tmp = exp_config.modify_GT_Head_0(y_init_tmp)
                    class_weights_head0 = utils.utils.get_class_weights(gt=y_init_tmp, verbose=verbose)

            

            logging.info('Loading logits from frozen network for the incremental data.')
            frozen_model_incremental_data_logits_filename = kwargs.get('frozen_model_incremental_data_logits_filename',
                                                                       None)
            if frozen_model_incremental_data_logits_filename is None:
                raise AssertionError('frozen_model_incremental_data_logits_filename is missing.')
            frozen_logits = h5py.File(frozen_model_incremental_data_logits_filename, 'r')
            sm_logits_incremental_frozen = frozen_logits['logits'] #contains all train_inds from exp_config.increment1()

    nlabels = exp_config_current.nlabels

    # Compute weight for each class based on training sample label count
    class_weights = utils.utils.get_class_weights(gt=y_train2D, verbose=verbose)


    ### Setup parameters for data generators for the graph
    seed_tf = 19
    channel_dim_exist = False
    key_specific_params = {'masks': {'interpolation': 0}, 'st_head0':{'channel_dim_exist': True}}  # make GT annotation augmentation interpolated with nearest neighbor
    data_augment_config = exp_config.data_augment_config


    if architecture_mods == 'incremental1':
        if exp_type in ['finetune']:
            keys_to_augment = ['images', 'masks']
            # data_train = {'images': x_train2D, 'masks': y_train2D, 'i': np.arange(len(x_train2D))}  # masks for newest 'head'
            data_train = {'images': x_train2D, 'masks': y_train2D, 'st_head0':x_train2D, 'i': np.arange(len(x_train2D))} #st_head0 is just a dummy
        else:
            keys_to_augment = ['images', 'masks', 'st_head0']
            data_train = {'images': x_train2D, 'masks': y_train2D, 'st_head0': sm_logits_incremental_frozen, 'i': np.arange(len(x_train2D))} #masks for newest 'head'
        generator_train = utils.image.DataGenerator(data=data_train, data_augment_config=data_augment_config,
                                                    keys_to_augment=keys_to_augment,
                                                    key_specific_params=key_specific_params, shuffle=True,
                                                    prng=exp_config.seed_keeper.prng, name='generator_train')
        data_val = {'images': x_validation2D, 'masks': y_validation2D, 'st_head0': y_validation2D_head0,'i': np.arange(len(x_validation2D))}
        generator_val = utils.image.DataGenerator(data=data_val, shuffle=True, name='generator_val')
        if exp_type in ['aeiseg', 'coriseg', 'reseg']:

            data_exemplar = {'images': x_train2D_exemplar, 'masks': y_train2D_exemplar , 'st_head0': sm_logits_exemplar_frozen, 'i': np.arange(len(x_train2D_exemplar))}
            generator_exemplar = utils.image.DataGenerator(data=data_exemplar, shuffle=True, data_augment_config=data_augment_config,
                                                           keys_to_augment=['images', 'masks', 'st_head0'],
                                                           prng=exp_config.seed_keeper.prng,
                                                           key_specific_params=key_specific_params, name='generator_exemplar')

    else:
        data_train = {'images': x_train2D, 'masks': y_train2D, 'i': np.arange(len(x_train2D))}
        data_val = {'images': x_validation2D, 'masks': y_validation2D,'i': np.arange(len(x_validation2D))}
        generator_val = utils.image.DataGenerator(data=data_val, shuffle=True, name='generator_val')
        generator_train = utils.image.DataGenerator(data=data_train, data_augment_config=data_augment_config,
                                                    keys_to_augment=['images', 'masks'],
                                                    key_specific_params=key_specific_params, shuffle=True,
                                                    prng=exp_config.seed_keeper.prng, name='generator_train')


    # build model into the default graph
    g = tf.Graph()
    with g.as_default():
        tf.random.set_random_seed(seed=seed_tf)

        x_shape = list(exp_config_current.image_size)
        y_shape = list(exp_config_current.image_size)
        x_dtype = tf.float32
        y_dtype = tf.uint8

        if architecture_mods == 'incremental1':
            if exp_type in ['finetune']:
                st_head0_shape = x_shape #surrogate targets for head 0
            else:    
                st_head0_shape = x_shape + [nlabels_head0] #surrogate targets for head 0
            st_head0_dtype = tf.float32
            dataset_train = tf.data.Dataset.from_generator(lambda: generator_train,
                                                           output_shapes={'images': x_shape, 'masks': y_shape, 'st_head0': st_head0_shape, 'i': []},
                                                           output_types={'images': x_dtype, 'masks': y_dtype, 'st_head0': st_head0_dtype,'i': tf.float32})

        else:
            dataset_train = tf.data.Dataset.from_generator(lambda: generator_train,
                                                           output_shapes={'images': x_shape, 'masks': y_shape, 'i': []},
                                                           output_types={'images': x_dtype, 'masks': y_dtype, 'i': tf.float32})


        dataset_train = dataset_train.repeat(count=exp_config.max_epochs)
        dataset_train = dataset_train.batch(exp_config_current.batch_size, drop_remainder=True)
        dataset_train = dataset_train.prefetch(buffer_size=exp_config_current.batch_size)
        # training set iterator
        iterator_train = dataset_train.make_initializable_iterator()

        if architecture_mods == 'incremental1' and exp_type in ['aeiseg', 'coriseg', 'reseg']:
            x_exemplar_shape = list(exp_config_current.image_size)
            x_exemplar_dtype = tf.float32
            st_head0_shape = x_exemplar_shape + [nlabels_head0]
            st_head0_dtype = tf.float32


            dataset_exemplar = tf.data.Dataset.from_generator(lambda: generator_exemplar,
                                                           output_shapes={'images': x_shape, 'st_head0': st_head0_shape, 'i': [], 'masks': y_shape},
                                                           output_types={'images': x_exemplar_dtype, 'st_head0': st_head0_dtype, 'i': tf.float32, 'masks': y_dtype})
            dataset_exemplar = dataset_exemplar.repeat(count=-1) #indefinitely repeat since we rely on new dataset epochs
            dataset_exemplar = dataset_exemplar.batch(exp_config_current.batch_size, drop_remainder=True)
            dataset_exemplar = dataset_exemplar.prefetch(buffer_size=exp_config_current.batch_size)
            # exemplar set iterator
            iterator_exemplar = dataset_exemplar.make_initializable_iterator()

        if architecture_mods == 'incremental1':
            val_head0_dtype = tf.float32
            dataset_val = tf.data.Dataset.from_generator(lambda: generator_val,
                                                           output_shapes={'images': x_shape, 'masks': y_shape, 'st_head0': y_shape, 'i': []},
                                                           output_types={'images': x_dtype, 'masks': y_dtype, 'st_head0': val_head0_dtype,'i': tf.float32})
        else:
            dataset_val = tf.data.Dataset.from_generator(lambda: generator_val,
                                                           output_shapes={'images': x_shape, 'masks': y_shape, 'i': []},
                                                           output_types={'images': x_dtype, 'masks': y_dtype, 'i': tf.float32})
        dataset_val = dataset_val.repeat(count=1)
        dataset_val = dataset_val.batch(exp_config_current.batch_size, drop_remainder=True)
        dataset_val = dataset_val.prefetch(buffer_size=exp_config_current.batch_size)
        # validation set iterator
        iterator_val = dataset_val.make_initializable_iterator()

        handle_iterator = tf.placeholder(tf.string, shape=[], name='generic_iterator_handle')
        iterator = tf.data.Iterator.from_string_handle(handle_iterator, output_types=dataset_train.output_types)  # need to ignore output_shapes since we need to do a trick with validation GT for head0
        batch_op = iterator.get_next()


        #tf.reshape() ops are necessary to break ambiguity for the rest of the graph.
        x_batch_pl = tf.reshape(batch_op['images'], shape=[exp_config_current.batch_size] + x_shape + [num_input_channels])
        y_batch_pl = tf.reshape(batch_op['masks'], shape=[exp_config_current.batch_size] + y_shape)
        y_batch_onehot = tf.one_hot(y_batch_pl, depth=nlabels)
        if architecture_mods == 'incremental1':
            st_pl_head0 = tf.reshape(batch_op['st_head0'], shape=[exp_config_current.batch_size] + y_shape + [nlabels_head0]) #soft target placeholder
            y_val_pl_head0 = tf.reshape(tf.cast(batch_op['st_head0'], dtype=tf.uint8), shape=[exp_config_current.batch_size] + y_shape)  #unfortunately feedable tf.dataset iterator requires same output convention across different generators, so GT for validation of classes of head0 go here.
            y_val_head0_onehot = tf.one_hot(y_val_pl_head0, depth=nlabels_head0) #convert y_validation set for head0 to onehot coding

        # Create placeholders for all necessary data input to TF Graph
        training_pl = tf.placeholder(tf.bool, shape=[], name='is_training_flag')
        learning_rate_pl = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        dropout_rate_pl = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_rate')

        # Define inference graph operation
        model_obj = exp_config.model_class(images=x_batch_pl, nlabels=nlabels,
                                           num_filters_first_layer=exp_config.num_filters_first_layer,
                                           training=training_pl, dropout_rate=dropout_rate_pl,
                                           architecture_mods=architecture_mods)
        if architecture_mods == 'incremental0':
            logits = model_obj.new_head(numClasses=exp_config_current.nlabels, head_name='Head_0')
            if hasattr(model_obj, 'uses_keras_layers'):
                if model_obj.uses_keras_layers:
                    model_obj.model = tf.keras.Model(inputs=model_obj.krs_inputs, outputs=logits)
                    model_obj.body_update_ops += [it for it in model_obj.model.updates if model_obj.network_scope in
                                                  it.name and it not in model_obj.body_update_ops]
                    model_obj.additional_update_ops['Head_0'] += [it for it in model_obj.model.updates
                                                                  if 'Head_0' in it.name and
                                                                  it not in model_obj.additional_update_ops['Head_0']]
        elif architecture_mods == 'incremental1':
            logits_head0 = model_obj.new_head(numClasses=exp_config.nlabels, head_name='Head_0')
            logits = model_obj.new_head(numClasses=exp_config_current.nlabels, head_name='Head_1')
            if hasattr(model_obj, 'uses_keras_layers'):
                if model_obj.uses_keras_layers:
                    model_obj.model = tf.keras.Model(inputs=model_obj.krs_inputs, outputs=[logits_head0, logits])
                    model_obj.body_update_ops += [it for it in model_obj.model.updates if model_obj.network_scope in
                                                  it.name and it not in model_obj.body_update_ops]
                    model_obj.additional_update_ops['Head_0'] += [it for it in model_obj.model.updates if 'Head_0' in
                                                                  it.name and it not in
                                                                  model_obj.additional_update_ops['Head_0']]
                    model_obj.additional_update_ops['Head_1'] += [it for it in model_obj.model.updates if 'Head_1' in
                                                                  it.name and it not in
                                                                  model_obj.additional_update_ops['Head_1']]

        summaries = tfboard.Summary()
        # Define histogram ops for tensorboard to track certain layer weights
        if hasattr(model_obj, 'uses_keras_layers'):
            if model_obj.uses_keras_layers:
                BN_tensors = [it for it in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'BN' in it.name]
        else:
            BN_tensors = [it for it in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'bn' in it.name]
        for it in BN_tensors:
            summaries.add_histogram(key='hist_' + it.name[:-2], val=it, category='histogram')

        # Define the logits loss graph op(s)
        loss_classification = wrappers.losses.weighted_cross_entropy_loss(logits=logits,
                                                                          labels=y_batch_onehot,
                                                                          class_weights=class_weights)


        loss_total = loss_classification

        
        if architecture_mods == 'incremental1' and exp_type in ['lwfseg', 'aeiseg', 'coriseg', 'reseg']:
            #compute fuzzy cross-entropy loss
            loss_distillation = wrappers.losses.weighted_cross_entropy_loss(logits=logits_head0,
                                                                              labels=st_pl_head0,
                                                                              class_weights=class_weights_head0)

            logging.info('New class weights: %s, old class weights: %s' % (str(class_weights),
                                                                                str(class_weights_head0)))
            loss_total = loss_total + loss_distillation
            summaries.add_scalar(val=loss_distillation, key='loss_distillation', category='Head_0')

        if architecture_mods == 'incremental0':
            summaries.add_scalar(val=loss_classification, key='loss_classification', category='Head_0')
            summaries.add_scalar(val=loss_total, key='loss_training', category='Head_0')
        elif architecture_mods == 'incremental1':
            summaries.add_scalar(val=loss_classification, key='loss_classification', category='Head_1')
            summaries.add_scalar(val=loss_total, key='loss_training', category='Head_1')
        else:
            print('For more increments, adapt the pipeline.')

        # Define the backprop op.
        if architecture_mods == 'incremental0':
            list_vars_head0 = model_obj.body_global_variables + model_obj.additional_global_variables['Head_0']
            list_ops_head0 = model_obj.body_update_ops + model_obj.additional_update_ops['Head_0']
            with tf.control_dependencies(list_ops_head0):
                train_op = exp_config_current.optimizer_handle(learning_rate=learning_rate_pl).minimize(loss_total, var_list=list_vars_head0)

        elif architecture_mods == 'incremental1':
            list_vars_head0 = model_obj.body_global_variables + model_obj.additional_global_variables['Head_0']
            list_ops_head0 = model_obj.body_update_ops + model_obj.additional_update_ops['Head_0']
            if exp_type in ['lwfseg', 'aeiseg', 'coriseg', 'reseg']:
                with tf.control_dependencies(list_ops_head0):
                    train_op_distillation = exp_config.optimizer_handle(learning_rate=learning_rate_pl).minimize(loss_distillation, var_list=list_vars_head0)

            list_vars_head1 = model_obj.body_global_variables + model_obj.additional_global_variables['Head_1']
            list_ops_head1 = model_obj.body_update_ops + model_obj.additional_update_ops['Head_1']
            with tf.control_dependencies(list_ops_head1):
                train_op_classification = exp_config_current.optimizer_handle(learning_rate=learning_rate_pl).minimize(loss_classification, var_list=list_vars_head1)

        # Define op to evaluate input im to y_batch
        eval_fn = lambda logits, onehot_labels, class_weights: [wrappers.losses.weighted_cross_entropy_loss(logits=logits, labels=onehot_labels, class_weights=class_weights),
                                                                wrappers.losses.weighted_soft_dice_score(logits=logits, labels=onehot_labels, class_weights=class_weights, channels_to_ignore=[0])]
        eval_loss = eval_fn(logits=logits, onehot_labels=y_batch_onehot, class_weights=class_weights)

        if architecture_mods == 'incremental1':
            eval_loss_head0 = eval_fn(logits=logits_head0, onehot_labels=y_batch_onehot, class_weights=class_weights_head0)
            eval_val_loss_head0 = eval_fn(logits=logits_head0, onehot_labels=y_val_head0_onehot, class_weights=class_weights_head0)


        ## Create image summaries for prediction outputs and GT
        idx_to_print = 0
        summaries.add_intensity_map(val=x_batch_pl[idx_to_print, ..., 0], key='image', category='all')
        summaries.add_intensity_map(val=x_batch_pl[idx_to_print, ..., 0], key='exemplar_image', category='exemplar')
        if architecture_mods == 'incremental0':
            summaries.add_label_map(val=y_batch_pl[idx_to_print], key='GT_Head0', category='Head_0', nlabels=exp_config_current.nlabels)
            summaries.add_label_map(val=tf.argmax(tf.nn.softmax(logits[idx_to_print], dim=-1), axis=-1), key='pred_Head0', category='Head_0', nlabels=exp_config_current.nlabels)
            for i in range(1,exp_config_current.nlabels):
                summaries.add_intensity_map(val=logits[idx_to_print, ..., i], key='logits_ch_'+str(i)+'_Head_0', category='Head_0')
        elif architecture_mods == 'incremental1':
            summaries.add_label_map(val=y_batch_pl[idx_to_print], key='GT_Head1', category='Head_1', nlabels=exp_config_current.nlabels)
            summaries.add_label_map(val=tf.argmax(tf.nn.softmax(logits[idx_to_print], dim=-1), axis=-1), key='pred_Head1', category='Head_1', nlabels=exp_config_current.nlabels)
            for i in range(1,exp_config_current.nlabels):
                summaries.add_intensity_map(val=logits[idx_to_print, ..., i], key='logits_ch_'+str(i)+'_Head_1', category='Head_1')
            summaries.add_label_map(val=y_batch_pl[idx_to_print], key='GT_Head0_exemplar', category='exemplar', nlabels=exp_config.nlabels)
            summaries.add_label_map(val=tf.argmax(tf.nn.softmax(logits_head0[idx_to_print], dim=-1), axis=-1), key='pred_Head0', category='Head_0', nlabels=exp_config.nlabels)
            for i in range(1,exp_config.nlabels):
                summaries.add_intensity_map(val=logits_head0[idx_to_print, ..., i], key='logits_ch_'+str(i)+'_Head_0', category='Head_0')
                if exp_type not in ['finetune']:
                    summaries.add_intensity_map(val=st_pl_head0[idx_to_print, ..., i], key='st_logits_ch_' + str(i) + '_Head_0', category='Head_0')
            #validation set
            summaries.add_intensity_map(val=x_batch_pl[idx_to_print, ..., 0], key='val_image', category='images_validation')
            summaries.add_label_map(val=y_val_pl_head0[idx_to_print], key='val_Head0_GT', category='images_validation', nlabels=exp_config.nlabels)
            summaries.add_label_map(val=y_batch_pl[idx_to_print], key='val_Head1_GT', category='images_validation', nlabels=exp_config_current.nlabels)
            summaries.add_label_map(val=tf.argmax(tf.nn.softmax(logits_head0[idx_to_print], dim=-1), axis=-1), key='val_pred_Head0', category='images_validation', nlabels=exp_config.nlabels)
            summaries.add_label_map(val=tf.argmax(tf.nn.softmax(logits[idx_to_print], dim=-1), axis=-1), key='val_pred_Head1', category='images_validation', nlabels=exp_config_current.nlabels)
            for i in range(1,exp_config_current.nlabels):
                summaries.add_intensity_map(val=logits[idx_to_print, ..., i], key='val_logits_ch_'+str(i)+'_Head_1', category='images_validation')
            for i in range(1,exp_config.nlabels):
                summaries.add_intensity_map(val=logits_head0[idx_to_print, ..., i], key='val_logits_ch_'+str(i)+'_Head_0', category='images_validation')
        else:
            print('Missing implementation.')

        
        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        max_to_keep = 2
        if architecture_mods == 'incremental1' and not continue_run: #not going to load Head_1 weights
            var_list = list_vars_head0
            if verbose:
                logging.info('Variable list from body and Head 0:\n%s' % (str(var_list)))
            saver_restore = tf.train.Saver(var_list=var_list, max_to_keep=max_to_keep, name='saver_body_and_Head_0')
        else:
            saver_restore = tf.train.Saver(max_to_keep=max_to_keep, name='saver_body_and_head_s_')
        saver = tf.train.Saver(max_to_keep=max_to_keep, name='saver_body_and_head_s_')
        saver_best_dice = tf.train.Saver()

        if verbose:
            # print the trainable variables
            tf.contrib.slim.model_analyzer.analyze_vars(
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), print_info=True)

        # Create a session for running Ops on the Graph.
        config = tf.ConfigProto()
        # Do not assign whole gpu memory, just use it on the go
        config.gpu_options.allow_growth = True
        # If a operation is not define it the default device, let it execute in another.
        config.allow_soft_placement = True
        config.gpu_options.per_process_gpu_memory_fraction = 1.
        sess = tf.Session(config=config)



        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        # Define any additional data to be exported to Tensorboard during training.
        val_error_ = tf.placeholder(tf.float32, shape=[], name='val_error')
        summaries.add_scalar(val=val_error_, key='val_error', category='validation')
        
        val_dice_ = tf.placeholder(tf.float32, shape=[], name='val_dice')
        summaries.add_scalar(val=val_dice_, key='validation_dice', category='validation')
        

        summaries.add_scalar(val=eval_loss[1], key='training_dice', category='all')
        if architecture_mods == 'incremental1': #for Head_0 when doing incremental training.
            val_error_head0_ = tf.placeholder(tf.float32, shape=[], name='val_error_Head_0')
            summaries.add_scalar(val=val_error_head0_, key='validation_loss_Head_0', category='validation_head0')

            val_dice_head0_ = tf.placeholder(tf.float32, shape=[], name='val_dice_Head_0')
            summaries.add_scalar(val=val_dice_head0_, key='validation_dice_Head_0', category='validation_head0')
            
            val_error_new_ = tf.placeholder(tf.float32, shape=[], name='val_error_Head_1')
            summaries.add_scalar(val=val_error_new_, key='validation_loss_Head_1', category='validation_head1')
            
            val_dice_new_ = tf.placeholder(tf.float32, shape=[], name='val_dice_Head_1')
            summaries.add_scalar(val=val_dice_new_, key='validation_dice_Head_1', category='validation_head1')
            

            if exp_type in ['aeiseg', 'coriseg', 'reseg']:
                summaries.add_scalar(val=eval_loss_head0[0], key='exemplar_loss_Head_0', category='exemplar')
                summaries.add_scalar(val=eval_loss_head0[1], key='exemplar_dice_Head_0', category='exemplar')


        hist_summaries = summaries.compile_category('histogram')
        val_summary = summaries.compile_category(category='validation')
        if architecture_mods == 'incremental0':
            summaries_head0 = summaries.compile_category(category=['all', 'Head_0'])
        elif architecture_mods == 'incremental1':
            summaries_inc_data = summaries.compile_category(category=['all', 'Head_0', 'Head_1'])
            summaries_exemplar = summaries.compile_category(category=['exemplar'])
            val_head0_summary = summaries.compile_category(category='validation_head0')
            val_head1_summary = summaries.compile_category(category='validation_head1')
            val_im_summary = summaries.compile_category(category='images_validation')

        # Run the Op to initialize the variables.
        sess.run(init)
        handle_training, handle_val = sess.run([iterator_train.string_handle(), iterator_val.string_handle()])
        if architecture_mods == 'incremental1' and exp_type in ['aeiseg', 'coriseg', 'reseg']:
            handle_exemplar = sess.run(iterator_exemplar.string_handle())


        if continue_run or architecture_mods == 'incremental1':
            # Restore session
            if verbose:
                logging.info('Printing tensors in checkpoint file %s' % (str(init_checkpoint_path)))
                chkp.print_tensors_in_checkpoint_file(init_checkpoint_path, tensor_name='', all_tensors=False,
                                                      all_tensor_names=True)
            saver_restore.restore(sess, init_checkpoint_path)

        step = init_step
        if step != 0:
            logging.info('%d more steps to train.' % (exp_config.max_steps - step))

        best_dice = 0
        curr_lr = exp_config.learning_rate

        if architecture_mods == 'incremental0' or exp_type in ['finetune','lwfseg']:
            num_total_iterations = int(np.ceil(x_train2D.shape[0] / exp_config_current.batch_size))
        elif architecture_mods == 'incremental1':
            num_total_iterations = int(np.max((np.ceil(x_train2D.shape[0] / exp_config_current.batch_size),
                                               np.ceil(x_train2D_exemplar.shape[0] / exp_config.batch_size))))
        logging.info('There are %i steps in each epoch.' % num_total_iterations)
        ex_counter, new_counter = 0, 0

        # Start Training
        sess.run(iterator_train.initializer)
        if architecture_mods == 'incremental1' and exp_type in ['aeiseg', 'coriseg', 'reseg']: # initialize iterators for exemplar images and soft targets
            sess.run(iterator_exemplar.initializer)
        for epoch in range(exp_config.max_epochs):

            logging.info('EPOCH %d' % epoch)

            # Train for each minibatch of data
            for epoch_step_count in range(num_total_iterations):

                if architecture_mods == 'incremental0':
                    ds_iterator_handle = handle_training
                elif architecture_mods == 'incremental1':
                    if seed_keeper.prng.rand() > train_ratio_exemplar_vs_new:
                        alt_ind = 1
                        new_counter += 1
                    else:
                        alt_ind = 0
                        ex_counter += 1

                    if alt_ind == 0: #exemplar_data
                        ds_iterator_handle = handle_exemplar
                    else: #new data
                        ds_iterator_handle = handle_training

                start_time = time.time()

                feed_dict = {
                    handle_iterator: ds_iterator_handle, #has images, masks, and soft targets
                    learning_rate_pl: curr_lr,
                    training_pl: True,
                    dropout_rate_pl: exp_config_current.dropout_rate
                }

                # Write the summaries and print an overview fairly often.
                if architecture_mods == 'incremental0':
                    if (step + 1) % exp_config.train_eval_frequency == 0:
                        logging.info('E: %d, Step %d: loss = %.2f (%.3f sec)' % (epoch, step, loss_value, duration))
                        # Update the events file.
                        _, loss_value, summary_str, [train_loss, train_dice], hist_summaries_msg = sess.run([train_op, loss_total, summaries_head0, eval_loss, hist_summaries], feed_dict=feed_dict)
                        logging.info('Training Data Eval:\nAverage loss: %0.04f, average dice: %0.04f' % (train_loss, train_dice))
                        
                        summary_writer.add_summary(summary_str, step)

                        ## BN histogram summary
                        summary_writer.add_summary(hist_summaries_msg, step)
                        # Update the events file.
                        summary_writer.flush()
                    else:
                        _, loss_value = sess.run([train_op, loss_total], feed_dict=feed_dict)

                elif architecture_mods == 'incremental1':
                    if alt_ind == 0: # old/exemplar data
                        if ex_counter % exp_config.train_eval_frequency == 0:
                            _, loss_value_distillation, summary_str, [train_loss_ex, train_dice_ex] = sess.run([train_op_distillation, loss_distillation, summaries_exemplar, eval_loss_head0],feed_dict=feed_dict)  # train_op_distillation uses frozen logits placeholder
                            logging.info('E: %d, Step %d: Exemplar data: losses (distillation) = %.3f (%.3f sec)' % (epoch, step, loss_value_distillation, duration))
                            logging.info('Exemplar Classes; training Data Eval:\nAverage loss: %0.04f, average dice: %0.04f' % (train_loss_ex, train_dice_ex))
                            summary_writer.add_summary(summary_str, step)
                        else:
                            _, loss_value_distillation = sess.run([train_op_distillation, loss_distillation], feed_dict=feed_dict)  # train_op_distillation uses frozen logits placeholder
                    elif alt_ind == 1:   # new data
                        if new_counter % exp_config.train_eval_frequency == 0:  # new data
                            if exp_type in ['lwfseg', 'aeiseg', 'coriseg', 'reseg']:
                                _, _, loss_value_classification, loss_value_distillation, summary_str, hist_summaries_msg, [train_loss_new, train_dice_new] = sess.run(
                                    [train_op_classification, train_op_distillation, loss_classification, loss_distillation, summaries_inc_data, hist_summaries, eval_loss], feed_dict=feed_dict)
                                logging.info('E: %d, Step %d: New data: losses (classification) = %.3f, (distillation) = %.3f (%.3f sec)' % (epoch, step, loss_value_classification, loss_value_distillation, duration))
                                ## histogram summary
                                summary_writer.add_summary(hist_summaries_msg, step)
                            else:  # finetuning
                                _, loss_value_classification, summary_str, [train_loss_new, train_dice_new] = sess.run([train_op_classification, loss_classification, summaries_inc_data, eval_loss],
                                                                        feed_dict=feed_dict)
                                logging.info('E: %d, Step %d: New data: losses (classification) = %.3f (%.3f sec)' % (epoch, step, loss_value_classification, duration))

                            logging.info('New Classes; training Data Eval:\nAverage loss: %0.04f, average dice: %0.04f' % (train_loss_new, train_dice_new))
                            
                            summary_writer.add_summary(summary_str, step)                            
                            summary_writer.flush()
                        else:
                            if exp_type in ['lwfseg', 'aeiseg', 'coriseg', 'reseg']:
                                _, _, loss_value_classification, loss_value_distillation = sess.run(
                                    [train_op_classification, train_op_distillation, loss_classification, loss_distillation], feed_dict=feed_dict)
                            else: #finetuning
                                _, loss_value_classification = sess.run([train_op_classification, loss_classification], feed_dict=feed_dict)

                    # Update the events file.
                    summary_writer.flush()

                duration = time.time() - start_time

                if (step + 1) % exp_config.val_eval_frequency == 0:
                    checkpoint_file = os.path.join(log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)

                # Evaluate validation set
                if architecture_mods == 'incremental0':
                    if (step + 1) % exp_config.val_eval_frequency == 0:


                        # Evaluate against the validation set (note that if you want to have different minibatch sizes, you'll need separate placeholders)
                        logging.info('Validation Data Eval:')
                        sess.run(iterator_val.initializer)
                        feed_dict = {
                            handle_iterator: handle_val,  # has x_validation2D and y_validation2D
                            training_pl: False,
                            dropout_rate_pl: 0.0}
                        num_eval_batches, eval_loss_tmp, eval_dice_tmp = 0, 0, 0
                        while True:
                            try:
                                loss_tmp, dice_tmp = sess.run(eval_loss, feed_dict=feed_dict)
                                num_eval_batches += 1
                                eval_loss_tmp += loss_tmp
                                eval_dice_tmp += dice_tmp
                            except tf.errors.OutOfRangeError:
                                break
                        avg_eval_loss = eval_loss_tmp / num_eval_batches
                        avg_eval_dice = eval_dice_tmp / num_eval_batches
                        logging.info('\nValidation set:\nAverage loss: %0.03f, average dice: %0.03f' % (avg_eval_loss, avg_eval_dice))

                        val_summary_msg = sess.run(val_summary, feed_dict={val_error_: avg_eval_loss, val_dice_: avg_eval_dice})
                        summary_writer.add_summary(val_summary_msg, step)
                        # Update the events file.
                        summary_writer.flush()
                        if avg_eval_dice > best_dice:
                            best_dice = avg_eval_dice
                            best_file = os.path.join(
                                log_dir, 'model_best_dice.ckpt')
                            saver_best_dice.save(sess, best_file, global_step=step)
                            logging.info('Found new best dice on validation set! - %f -  Saving model_best_dice.ckpt' % avg_eval_dice)

                elif architecture_mods == 'incremental1':
                    if alt_ind == 1 and new_counter % exp_config.val_eval_frequency == 0: #Evaluate for both Head_0 and Head_1, respectively

                        logging.info('Validation Data Eval (Both Heads):')
                        sess.run(iterator_val.initializer)
                        feed_dict = {
                            handle_iterator: handle_val,  # has x_validation2D and stacked_y_validation2D[1]
                            training_pl: False,
                            dropout_rate_pl: 0.0}
                        num_eval_batches = 0
                        eval_loss_tmp, eval_dice_tmp = 0, 0
                        eval_loss_head0_tmp, eval_dice_head0_tmp = 0, 0
                        while True:
                            try:
                                if num_eval_batches == 0:
                                    val_im_summary_str, [loss_tmp, dice_tmp], [loss_head0_tmp, dice_head0_tmp] = sess.run([val_im_summary, eval_loss, eval_val_loss_head0], feed_dict=feed_dict)
                                else:
                                    [loss_tmp, dice_tmp], [loss_head0_tmp, dice_head0_tmp] = sess.run([eval_loss, eval_val_loss_head0], feed_dict=feed_dict)
                                num_eval_batches += 1
                                eval_loss_tmp += loss_tmp
                                eval_dice_tmp += dice_tmp
                                eval_loss_head0_tmp += loss_head0_tmp
                                eval_dice_head0_tmp += dice_head0_tmp
                            except tf.errors.OutOfRangeError:
                                break
                        val_loss = eval_loss_tmp / num_eval_batches
                        val_dice = eval_dice_tmp / num_eval_batches
                        val_loss_head0 = eval_loss_head0_tmp / num_eval_batches
                        val_dice_head0 = eval_dice_head0_tmp / num_eval_batches

                        cum_val_loss = (val_loss + val_loss_head0) /2.
                        cum_dice = (val_dice + val_dice_head0) /2.

                        summary_writer.add_summary(val_im_summary_str, step)

                        val_summary_msg = sess.run(val_summary,feed_dict={val_error_: cum_val_loss, val_dice_: cum_dice})
                        summary_writer.add_summary(val_summary_msg, step)
                        val_summary_msg = sess.run(val_head0_summary, feed_dict={val_error_head0_: val_loss_head0,
                                                                                   val_dice_head0_: val_dice_head0})
                        summary_writer.add_summary(val_summary_msg, step)
                        val_summary_msg = sess.run(val_head1_summary,feed_dict={val_error_new_: val_loss, val_dice_new_: val_dice})
                        summary_writer.add_summary(val_summary_msg, step)
                        # Update the events file.
                        summary_writer.flush()

                        logging.info('\nValidation set:\nAverage loss: %0.03f, average dice: %0.03f' % (cum_val_loss, cum_dice))
                        logging.info('\nHead0: loss: %0.03f, average dice: %0.03f' % (val_loss_head0, val_dice_head0))
                        logging.info('\nHead1: loss: %0.03f, average dice: %0.03f' % (val_loss, val_dice))

                        if cum_dice > best_dice:
                            best_dice = cum_dice
                            best_file = os.path.join(log_dir, 'model_best_dice.ckpt')
                            saver_best_dice.save(sess, best_file, global_step=step)
                            logging.info('Found new best dice on validation set! - %s, avg: %.3f -  Saving model_best_dice.ckpt'% (str([val_dice_head0, val_dice]), best_dice))

                # Single learning step successfully completed, we can now save the source code.
                if epoch == 0 and epoch_step_count == 0:
                    if not continue_run:
                        # Copy source code into output directory.
                        for i in exp_config.source_to_copy:
                            if additional_save_info is not None:
                                fid = open(os.path.join(log_dir_source, 'extra.sh'), 'w')
                                fid.write(str(additional_save_info))
                                fid.close()
                            if os.path.isdir(i):
                                external.utils.copy_dirs(src=i, dst=os.path.join(log_dir_source, i), verbose=False)
                            else:
                                shutil.copy2(i, log_dir_source)
                        print('Source files have been copied.')
                if step >= exp_config.max_steps:
                    sess.close()
                    g.finalize()
                    data.close()
                    logging.info(
                        '---------- End of Training reached. -----------')
                    return
                step += 1
        sess.close()
        g.finalize()
    data.close()
    if architecture_mods == 'incremental1' and exp_type in ['aeiseg', 'coriseg', 'reseg', 'lwfseg']:
        data_exemplar.close()
    logging.info('---------- End of Training reached. -----------')
    logging.info('Logdir was: %s' % (log_dir))




def exp_name(exp_config, **kwargs):

    init_training_data = kwargs.get('init_train_on_incremental', '')
    arch = kwargs.get('architecture', 'unknownArchitecture')
    try:
        scenario = exp_config.case_ind
        if scenario == 0:
            scenario = '_Case1'
        elif scenario == 1:
            scenario = '_Case2'
        elif scenario == 2:
            scenario = '_Case3'
        elif scenario == 3:
            scenario = '_Case4'
    except:
        scenario = ''
    try:
        modality_str = exp_config.modality_str_id
    except:
        modality_str = exp_config.modality_str
    return arch + '_wxent_{!s}_modality_{!s}_LR2S_resize{!s}_{!s}{!s}{!s}'.format(
        str(exp_config.label_name), str(modality_str), str(exp_config.image_size[0]),
        str(exp_config.target_resolution[0]), str(scenario), str(init_training_data))


def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # default_parent_dir = '/scratch_net/hartley_third/ozdemirf/savedModels'
    default_parent_dir = '/scratch_net/hartley_fourth/ozdemirf/savedModels'



    timestmp = ''

    experiment_configuration_default = 'incremental/unet2D_BN_wxent_SKI10_patches_1femur_2tibia' #'wxent_supraspinatus_LR2S'

    parser = argparse.ArgumentParser()
    # parser.add_argument("-g", "--gpu_list", type=str, help="list of GPUs to use", default=None)
    parser.add_argument("-c", "--config", type=str, help="name of the experiment configuration file, defaults to %s" % experiment_configuration_default,
                        default=experiment_configuration_default)
    parser.add_argument("-l", "--logdir", type=str,
                        help="directory to save model and export tensorboard under /scratch_net/hartley_second/ozdemirf/savedModels", default='')
    parser.add_argument("-s", "--compute_exemplar_samples", help="compute exemplar samples, defaults to False", action="store_true")
    parser.add_argument("--K_uncertain", type=int, help="#uncertain samples for exemplar set, defaults to exp_config", default=0)
    parser.add_argument("--k_rep", type=int, help="#representative samples for exemplar set, defaults to exp_config", default=0)
    parser.add_argument("-v", "--verbose", help="set verbosity, defaults to False", action="store_true")
    parser.add_argument("-d", "--debug_mode", help="set debug mode, defaults to False", action="store_true")
    parser.add_argument("-t", "--continue_run",
                        help="continue training in directory, defaults to False", action="store_true")
    parser.add_argument("-m", "--architecture_mods", type=str,
                        help="architecture modifications for training {'default', 'incremental0', 'incremental1'} defaults to incremental0", default='incremental0')
    parser.add_argument("-e", "--exp_type", type=str, help="options are finetune, lwfseg, aeiseg, coriseg, reseg default: coriseg", default='coriseg')
    parser.add_argument("-i", "--exp_scenario", type=str, help="options are '1', '2', '3', '4', default: '1'", default='1')
    parser.add_argument("--index_shuffle_seed_ID", type=str, help="options are '1', '2', '3', '4', '5' default: '1'", default='1')
    parser.add_argument("--uncertainty_method", type=str, help="options are mcdropout, BALD default: 'mcdropout", default='mcdropout')
    parser.add_argument("-n", "--init_train_on_incremental", help="runs initial training on incremental data (part of evaluation), defaults to False (init data)", action="store_true")

    args = parser.parse_args()

    exp_config_filename = args.config
    config_name = os.path.join('experiment_configurations', exp_config_filename + '.py')
    spec = importlib.util.spec_from_file_location(exp_config_filename, config_name)
    exp_config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp_config_module)
    exp_config = exp_config_module.ExperimentConfig()
    if args.K_uncertain != 0:
        exp_config.K_most_certain = args.K_uncertain
    if args.k_rep != 0:
        exp_config.k_representative = args.k_rep

    exp_config.set_initial_seed(seedID=args.index_shuffle_seed_ID)
    exp_config.set_case(caseID=args.exp_scenario)
    
    if _USING_SGE_GPU:
        logging.info('using gpu list: %s' % (str(os.environ['CUDA_VISIBLE_DEVICES'])))

    exp_config.model_class = architectures.unet_keras.UNet_keras

    architecture_mods = args.architecture_mods
    exp_type = args.exp_type
    if architecture_mods not in ['incremental0', 'incremental1']:
        raise AssertionError('Unknown architecture_mod %s provided.' % architecture_mods)
    if exp_type not in ['finetune', 'lwfseg', 'aeiseg', 'coriseg', 'reseg']:
        raise AssertionError('Unknown exp_type %s provided.' % exp_type)
    init_train_on_incremental = args.init_train_on_incremental
    if not (architecture_mods == 'incremental0') and init_train_on_incremental:
        raise AssertionError('init_train_on_incremental: %s, architecture_mods: %s' % (str(init_train_on_incremental),
                                                                                       str(architecture_mods)))
    exp_config.architecture_mods = architecture_mods
    uncertainty_method = args.uncertainty_method
    str_init_train_on_incremental = ''
    if init_train_on_incremental:
        if args.verbose:
            logging.info('init_train_on_incremental is Enabled.')
        exp_config.set_inds_to_incremental_in_init()
        str_init_train_on_incremental = '_trainOnIncrementalData'

    exp_config.experiment_name = exp_name(exp_config=exp_config, architecture='unet2d_dropout',
                                          init_train_on_incremental=str_init_train_on_incremental)

    if args.verbose:
        logging.info('Dataset Split ratio: %s' % (str(exp_config.ratio_train_valid_test)))
    if args.logdir == '':
        log_dir_tmp = os.path.join(default_parent_dir, timestmp + exp_config.experiment_name)
        continue_run = False
    else:
        log_dir_tmp = os.path.join(default_parent_dir, args.logdir)
        continue_run = args.continue_run

    initial_log_dir = log_dir_tmp
    exp_path = os.path.join(initial_log_dir, architecture_mods)
    frozen_model_incremental_data_logits_filename = os.path.join(exp_path, 'frozen_logits.hdf5')
    log_dir_tmp_inc = os.path.join(exp_path, exp_type)
    exemplar_data_path = os.path.join(log_dir_tmp_inc, 'comp_data')
    Kk_config = 'K'+str(exp_config.K_most_certain)+'_k'+str(exp_config.k_representative)
    if uncertainty_method == 'BALD':
        Kk_config += '_uncBALD'
    exemplar_info_filename = os.path.join(exemplar_data_path, 'exemplar_' + Kk_config + '.hdf5')
    if architecture_mods == 'incremental1':
        if not os.path.isdir(exp_path):
            logging.info('Creating directory %s' % (exp_path))
            os.mkdir(exp_path)
        if not os.path.isdir(log_dir_tmp_inc):
            os.mkdir(log_dir_tmp_inc)
        log_dir_tmp = os.path.join(log_dir_tmp_inc, Kk_config)
        if not os.path.isdir(log_dir_tmp):
            os.mkdir(log_dir_tmp)
        use_best_validation_dice = True #If false, it will load latest checkpoint of saved model
        if use_best_validation_dice:
            checkpoint_path = external.utils.get_latest_model_checkpoint_path(initial_log_dir, 'model_best_dice.ckpt')
        else:
            checkpoint_path = external.utils.get_latest_model_checkpoint_path(initial_log_dir, 'model.ckpt')
        if exp_type in ['lwfseg', 'aeiseg', 'coriseg', 'reseg']:
            if not os.path.isfile(frozen_model_incremental_data_logits_filename):
                logging.info('frozen logits do not exist. Will be computed now into: %s' %
                             str(frozen_model_incremental_data_logits_filename))
                exp_config_current = exp_config.Iteration1
                h5FileName = os.path.join(exp_config_current.dataset_parent_folder, exp_config_current.dataset)
                data = h5py.File(h5FileName, "r")
                inds_train = exp_config_current.inds_train
                x_train2D = data['images'][inds_train, ...]
                pre_inc_train.compute_logits(images=x_train2D, masks=None, indices=inds_train, exp_config=exp_config,
                               filename=frozen_model_incremental_data_logits_filename, checkpoint_path=checkpoint_path)
                data.close()

            if args.compute_exemplar_samples:
                logging.info('Going to compute exemplar samples for architecture_mod: %s, exp_type: %s. Save path:%s'
                             % (architecture_mods, exp_type, exemplar_info_filename))
                if not os.path.isdir(exemplar_data_path):
                    os.mkdir(exemplar_data_path)
                if exp_type == 'reseg':
                    #pick random samples to be exemplars
                    pre_inc_train.randomly_pick_exemplar_samples(exp_type=exp_type, exemplar_filepath=exemplar_data_path, exp_config=exp_config,
                                             checkpoint_path=checkpoint_path,exemplar_info_filename=exemplar_info_filename)
                else:
                    pre_inc_train.compute_exemplar_samples(exp_type=exp_type, uncertainty_filepath=exp_path,
                                             exemplar_filepath=exemplar_data_path, exp_config=exp_config,
                                             checkpoint_path=checkpoint_path, exemplar_info_filename=exemplar_info_filename,
                                             uncertainty_method=uncertainty_method)
                logging.info('Exemplar sample selection finished.')



    if args.verbose:
        if continue_run:
            logging.info('Training will continue.')
        else:
            logging.info('Training will start from scratch.')
        logging.info('additional_save_info: %s' % str(args))
        logging.info('experiment_name: %s' % str(exp_config.experiment_name))
        logging.info('Dataset name is: %s' % str(exp_config.dataset))
    # logging.info('using gpu list: %s' % (str(args.gpu_list)))

    logging.info('Logdir is: %s\ncontinue_training: %s' % (log_dir_tmp, str(continue_run)))

    train(exp_config=exp_config,
          continue_run=continue_run,
          log_dir=log_dir_tmp,
          verbose=args.verbose,
          _DEBUG=args.debug_mode,
          additional_save_info=args,
          architecture_mods = architecture_mods,
          exp_type=exp_type,
          initial_log_dir=initial_log_dir,
          frozen_model_incremental_data_logits_filename=frozen_model_incremental_data_logits_filename,
          exemplar_info_filename=exemplar_info_filename,
          init_train_on_incremental=init_train_on_incremental)



if __name__ == "__main__":
    # tf.app.run()
    main(0)
