########################
# Author:
# Firat Ozdemir (fozdemir@gmail.com), Copyright (R) 2018, ETH Zurich
########################

import os

_USING_SGE_GPU = False
if "CUDA_VISIBLE_DEVICES" in os.environ:
    _USING_SGE_GPU = True
    import matplotlib as mpl #overcome use of XServer for images
    mpl.use('Agg')
import numpy as np
import tensorflow as tf
import time

import logging
import utils.utils
import utils.image
import external.utils
import h5py


def randomly_pick_exemplar_samples(exp_type, exemplar_filepath, exp_config, checkpoint_path, verbose=True, **kwargs):
    if exp_type not in ['reseg']:
        raise AssertionError('Unexpected exp_type: %s' % (exp_type))
    fname_exemplar_samples = kwargs.get('exemplar_info_filename', os.path.join(exemplar_filepath, 'exemplar.hdf5'))
    init_train_on_incremental = kwargs.get('init_train_on_incremental', False)
    if os.path.isfile(fname_exemplar_samples):
        logging.info('fname_exemplar_samples exists: %s' % (str(fname_exemplar_samples)))
        return
    # Load dataset
    h5FileName = os.path.join(exp_config.dataset_parent_folder, exp_config.dataset)
    data = h5py.File(h5FileName, "r")
    inds_train = exp_config.inds_train
    x_train2D, y_train2D = data['images'][inds_train, ...], data['masks'][inds_train, ...]
    if init_train_on_incremental:
        y_train2D = exp_config.modify_GT_Head_1(y_train2D)
    else:
        y_train2D = exp_config.modify_GT_Head_0(y_train2D)
    class_weights = utils.utils.get_class_weights(gt=y_train2D)
    k = exp_config.k_representative #number of exemplar samples for each class.
    sample_has_lbl = check_if_lbl_exists(y=y_train2D)
    at_least_K_samples_exist = [np.sum(sample_has_lbl[i, ...]) < k for i in range(exp_config.nlabels)]  # check if there are enough samples with respective label
    if np.any(at_least_K_samples_exist):
        raise AssertionError('sample_has_lbl has samples with labels on less than %d: %s' % (k, str(at_least_K_samples_exist)))
    indices_for_argsort = [np.arange(len(inds_train))[sample_has_lbl[i, ...]] for i in range(exp_config.nlabels)]
    random_exemplars = [np.arange(len(inds)) for inds in indices_for_argsort]
    prng = exp_config.seed_keeper.prng
    set_k = []
    for i_lbl in range(exp_config.nlabels):
        prng.shuffle(random_exemplars[i_lbl])
        set_k.append(indices_for_argsort[i_lbl][random_exemplars[i_lbl][:k]]) #To be fair, we want to draw k samples randomly for each label, union set can be smaller than nlabels*k
    flat_index_list = np.unique(np.reshape(set_k, newshape=(-1)))
    # for each label, add the exemplar images / GT into an hdf5 file.
    images_exemplar = x_train2D[flat_index_list, ...]
    masks_exemplar = y_train2D[flat_index_list, ...]

    compute_logits(images=images_exemplar, masks=masks_exemplar, indices=flat_index_list,
                   filename=fname_exemplar_samples, exp_config=exp_config, checkpoint_path=checkpoint_path,
                   class_weights=class_weights)

def compute_exemplar_samples(exp_type, uncertainty_filepath, exemplar_filepath, exp_config, checkpoint_path, verbose=True, **kwargs):
    fname_exemplar_samples = kwargs.get('exemplar_info_filename', os.path.join(exemplar_filepath, 'exemplar.hdf5'))
    init_train_on_incremental = kwargs.get('init_train_on_incremental', False)
    uncertainty_method = kwargs.get('uncertainty_method', 'mcdropout')
    if os.path.isfile(fname_exemplar_samples):
        logging.info('fname_exemplar_samples exists: %s' % (str(fname_exemplar_samples)))
        return
    if exp_type not in ['aeiseg', 'coriseg']:
        raise AssertionError('Unexpected exp_type: %s' % (exp_type))
    K = exp_config.K_most_certain
    k = exp_config.k_representative
    num_mcdropout = exp_config.MCUncertainty_count
    logging.info('exp type: %s\n#MC dropout for uncertainty computation: %d\n#samples uncertainty: %d\n#samples representative: %d'
                 % (exp_type, num_mcdropout, K, k))
    if uncertainty_method == 'mcdropout':
        fname_uncertainty = os.path.join(uncertainty_filepath, 'uncertainty.hdf5')
    elif uncertainty_method == 'BALD':
        fname_uncertainty = os.path.join(uncertainty_filepath, 'uncertainty_BALD.hdf5')
    fname_image_descriptors = os.path.join(exemplar_filepath, 'image_descriptors.hdf5')
    fname_dist_mtx = os.path.join(exemplar_filepath, 'dist_mtx.hdf5')
    if uncertainty_method == 'mcdropout':
        fname_representative_set = os.path.join(exemplar_filepath, 'S_a_K'+str(K)+'_k'+str(k)+'.npz')
    elif uncertainty_method == 'BALD':
        fname_representative_set = os.path.join(exemplar_filepath, 'S_a_K' + str(K) + '_k' + str(k) + '_BALD.npz')
    #Load dataset
    h5FileName = os.path.join(exp_config.dataset_parent_folder, exp_config.dataset)
    data = h5py.File(h5FileName, "r")
    inds_train = exp_config.inds_train
    x_train2D, y_train2D = data['images'][inds_train, ...], data['masks'][inds_train, ...]
    if init_train_on_incremental:
        y_train2D = exp_config.modify_GT_Head_1(y_train2D)
    else:
        y_train2D = exp_config.modify_GT_Head_0(y_train2D)
    class_weights = utils.utils.get_class_weights(gt=y_train2D)

    if os.path.isfile(fname_uncertainty): #compute uncertainty values
        logging.info('Uncertainty file %s exists, skipping..' % (fname_uncertainty))
    else:
        compute_uncertainty(images=x_train2D, filename=fname_uncertainty, exp_config=exp_config, method=uncertainty_method,
                            dropout_rate=0.5, checkpoint_path=checkpoint_path, num_mcdropout=num_mcdropout)
    # find most certain K samples.
    hdf5_file = h5py.File(fname_uncertainty, 'r')
    uncertainty_scores = np.asarray(hdf5_file['uncertainty_scores'])
    sample_has_lbl = check_if_lbl_exists(y=y_train2D)
    at_least_K_samples_exist = [np.sum(sample_has_lbl[i, ...]) < K for i in range(exp_config.nlabels)] #check if there are enough samples with respective label
    if verbose:
        logging.info('shape of sample_has_lbl: %s\n shape of uncertainty_scores: %s' % (str(sample_has_lbl.shape), str(uncertainty_scores)))
    if np.any(at_least_K_samples_exist):
        raise AssertionError('sample_has_lbl has samples with labels on less than %d: %s' % (K, str(at_least_K_samples_exist)))
    indices_for_argsort = [np.arange(len(inds_train))[sample_has_lbl[i, ...]] for i in range(exp_config.nlabels)]
    uncertainty_scores_nolbl_samples_removed = [uncertainty_scores[sample_has_lbl[i, ...], i] for i in range(exp_config.nlabels)]
    argsort_uncertainty_scores_nolbl_samples_removed = [indices_for_argsort[i][np.argsort(uncertainty_scores_nolbl_samples_removed[i])] for i in range(exp_config.nlabels)]
    sorted_uncertainty_scores_nolbl_samples_removed = [uncertainty_scores[argsort_uncertainty_scores_nolbl_samples_removed[i], i] for i in range(exp_config.nlabels)]
    if verbose:
        logging.info('Available #samples for each label: %s' % (str([len(sorted_uncertainty_scores_nolbl_samples_removed[i])
                                                                     for i in range(exp_config.nlabels)])))
    inds_K_most_certain_samples = np.asarray([argsort_uncertainty_scores_nolbl_samples_removed[i][:K] for i in range(exp_config.nlabels)])
    flat_uncertainty_index_list = np.unique(np.reshape(inds_K_most_certain_samples, newshape=(-1)))

    if verbose:
        logging.info('Uncertainty scores for first 10 elements when GS lbl exists:\n%s' % (str([sorted_uncertainty_scores_nolbl_samples_removed[i][:10] for i in range(exp_config.nlabels)])))
    # #TODO: next, compute image descriptors/vgg16 tensors for these K samples for each label.
    if exp_type == 'aeiseg': #compute image descriptors
        if os.path.isfile(fname_image_descriptors):  # compute uncertainty values
            logging.info('image_descriptors file %s exists, skipping..' % (fname_image_descriptors))
        else:
            compute_image_descriptors(images=x_train2D, filename=fname_image_descriptors,
                                      exp_config=exp_config, checkpoint_path=checkpoint_path,
                                      uncertainty_scores=uncertainty_scores)
        if os.path.isfile(fname_dist_mtx):
            logging.info('fname_dist_mtx file %s exists, skipping..' % (fname_dist_mtx))
            ftmp = h5py.File(fname_dist_mtx, 'r')
            similarity_mtx = ftmp['similarity_mtx']
        else:
            similarity_mtx = compute_distance_matrix(filename=fname_image_descriptors, distance_metric='cosine_angle',
                                                     savename=fname_dist_mtx)
        if verbose:
            logging.info('similarity_mtx is in range: [%.5f, %.5f], shape: %s' % (np.min(similarity_mtx), np.max(similarity_mtx), str(np.shape(similarity_mtx))))
        similarity_mtx = similarity_mtx - np.min(similarity_mtx)
        similarity_mtx = similarity_mtx/np.max(similarity_mtx)
        dist_mtx = 1.0-(similarity_mtx)

    elif exp_type == 'coriseg': #compute vgg16 tensors

        if os.path.isfile(fname_dist_mtx):
            ftmp = h5py.File(fname_dist_mtx, 'r')
            dist_mtx = ftmp['dist_mtx'].value
        else:
            dist_mtx = compute_vgg19_tensor_distance(images=x_train2D, filename=fname_dist_mtx, exp_config=exp_config,
                uncertainty_scores=uncertainty_scores, distance_metric='MSE')
        if verbose:
            logging.info('distance_mtx is in range: [%.5f, %.5f], shape: %s' % (
            np.min(dist_mtx), np.max(dist_mtx), str(np.shape(dist_mtx))))
        dist_mtx = dist_mtx - np.min(dist_mtx)
        dist_mtx = dist_mtx / np.max(dist_mtx)

    #TODO: Since this if clause below should be fast, make it extracted for each experiment, as we may want to try different k values.
    if os.path.isfile(fname_representative_set):  # compute uncertainty values
        logging.info('fname_representative_set file %s exists, skipping..' % (fname_representative_set))
        tmp_dict = np.load(fname_representative_set)['arr_0'].item()
        set_k = tmp_dict['set_k']
        set_dist_over_time = tmp_dict['set_dist_over_time']
    else:
        set_k, set_dist_over_time = max_set_cover(distance_mtx=dist_mtx, inds_most_certain=inds_K_most_certain_samples, return_set_size=k)
        np.savez(fname_representative_set, {'set_k':set_k, 'set_dist_over_time':set_dist_over_time})

    flat_index_list = np.unique(np.reshape(set_k, newshape=(-1)))
    cond_rep_elm_not_in_unc = [elm_S_a not in flat_uncertainty_index_list for elm_S_a in flat_index_list]
    if np.any(cond_rep_elm_not_in_unc):
        raise AssertionError('representativeness set contains images of indices not from most uncertain indices. \nrep: %s, \nunc: %s\nElements are: %s'
                             % (str(flat_index_list), str(flat_uncertainty_index_list), str(flat_index_list[cond_rep_elm_not_in_unc])))
    #for each label, add the exemplar images / GT into an hdf5 file.
    images_exemplar = x_train2D[flat_index_list,...]
    masks_exemplar = y_train2D[flat_index_list,...]

    compute_logits(images=images_exemplar, masks=masks_exemplar, indices=flat_index_list,
                   filename=fname_exemplar_samples, exp_config=exp_config, checkpoint_path=checkpoint_path,
                   class_weights=class_weights, set_dist_over_time=set_dist_over_time)


def compute_logits(images, masks, indices, filename, exp_config, checkpoint_path, dropout_rate=0.0, verbose=True, **kwargs):
    if masks is None:
        class_weights = kwargs.get('class_weights', None)
    else:
        class_weights = kwargs.get('class_weights', utils.utils.get_class_weights(gt=masks))
    set_dist_over_time = kwargs.get('set_dist_over_time', None)
    hdf5_file = h5py.File(filename, "w")
    hdf5_file.create_dataset(name='indices', data=indices, dtype=np.int32)
    hdf5_file.create_dataset(name='class_weight', data=class_weights, dtype=np.float32)
    if set_dist_over_time is not None:
        hdf5_file.create_dataset(name='set_dist_over_time', data=set_dist_over_time, dtype=np.float32)

    if len(images.shape) == 3:  # single-channel input
        num_input_channels = 1
    elif len(images.shape) == 4:  # multi-channel input
        num_input_channels = images.shape[3]
    nlabels = exp_config.nlabels
    batch_size = exp_config.batch_size
    if len(images) < batch_size:
        batch_size = len(images)
    images_shp = images.shape

    data = dict()
    data['logits'] = hdf5_file.create_dataset('logits', shape=list(images_shp[:3]) + [nlabels], dtype=np.float32)
    data['images'] = hdf5_file.create_dataset('images', shape=list(images_shp), dtype=np.int32)
    if masks is not None:
        data['masks'] = hdf5_file.create_dataset('masks', shape=list(masks.shape), dtype=np.int32)

    if masks is None:
        data_dict = {'images': images}
    else:
        data_dict = {'images': images, 'masks': masks}
    key_specific_params = {'masks': {'interpolation': 0}}
    generator = utils.image.DataGenerator(data=data_dict, shuffle=False, key_specific_params=key_specific_params)

    total_time = 0
    start_time = time.time()

    g = tf.Graph()
    with g.as_default():

        x_shape = list(exp_config.image_size)
        y_shape = list(exp_config.image_size)
        x_dtype = tf.float32
        y_dtype = tf.uint8

        if masks is None:
            dataset = tf.data.Dataset.from_generator(lambda: generator,
                                                     output_shapes={'images': x_shape},
                                                     output_types={'images': x_dtype})
        else:
            dataset = tf.data.Dataset.from_generator(lambda: generator,
                                                     output_shapes={'images': x_shape, 'masks': y_shape},
                                                     output_types={'images': x_dtype, 'masks': y_dtype})

        dataset = dataset.repeat(count=1)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=batch_size)

        iterator = dataset.make_initializable_iterator()
        batch_op = iterator.get_next()

        if num_input_channels == 1:
            x_batch_pl = tf.expand_dims(batch_op['images'], axis=-1)
        else:
            x_batch_pl = batch_op['images']
        if masks is not None:
            y_batch_pl = batch_op['masks']
        # y_batch_pl = tf.reshape(batch_op['masks'], shape=[exp_config_current.batch_size] + y_shape)


        # Create placeholders for all necessary data input to TF Graph
        training_pl = tf.constant(False, dtype=tf.bool, shape=[], name='is_training_flag')
        dropout_rate_pl = tf.constant(dropout_rate, dtype=tf.float32, shape=[], name='dropout_rate')
        # Define inference graph operation
        model_obj = exp_config.model_class(images=x_batch_pl, nlabels=nlabels,
                                           num_filters_first_layer=exp_config.num_filters_first_layer,
                                           training=training_pl, dropout_rate=dropout_rate_pl,
                                           architecture_mods='incremental0')
        logits = model_obj.new_head(numClasses=exp_config.nlabels, head_name='Head_0')
        if hasattr(model_obj, 'uses_keras_layers'):
            if model_obj.uses_keras_layers:
                model_obj.model = tf.keras.Model(inputs=model_obj.krs_inputs, outputs=logits)

        sm_logits = tf.nn.softmax(logits)

        saver = tf.train.Saver(max_to_keep=1)

        init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
        config.allow_soft_placement = True  # If a operation is not define it the default device, let it execute in another.
        config.gpu_options.per_process_gpu_memory_fraction = 1.0
        sess = tf.Session(config=config)

        sess.run(init)

        # Restore model weights
        saver.restore(sess, checkpoint_path)

        sess.run(iterator.initializer)

        num_total_iterations = int(np.ceil(images_shp[0] / batch_size))
        if verbose:
            logging.info('There are %d images to be inferenced' % (images_shp[0]))

        train_index_counter = 0 #this is just a security measure to make sure generator indices match original dataset indices

        for ind in range(num_total_iterations):
            logging.info('Processing minibatch %i out of %i.' % (ind + 1, num_total_iterations))

            train_index_counter = train_index_counter + batch_size

            if masks is None:
                x_batch, np_sm_logits = sess.run([x_batch_pl, sm_logits])
            else:
                x_batch, y_batch, np_sm_logits = sess.run([x_batch_pl, y_batch_pl, sm_logits])

            counter_from = train_index_counter - batch_size
            counter_to = train_index_counter


            external.utils.write_range_to_hdf5(hdf5_data=data, list_name='images', data_list=np.squeeze(x_batch), dtype=np.float32,
                                counter_from=counter_from, counter_to=counter_to)
            external.utils.write_range_to_hdf5(hdf5_data=data, list_name='logits', data_list=np_sm_logits,
                                               dtype=np.float32, counter_from=counter_from, counter_to=counter_to)
            if masks is not None:
                external.utils.write_range_to_hdf5(hdf5_data=data, list_name='masks',
                                                   data_list=np.squeeze(y_batch), dtype=np.float32,
                                                   counter_from=counter_from, counter_to=counter_to)

            if ind != 0 and ind % 10 == 0:
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                start_time = time.time()
                logging.info('Evaluation of 10 minibatches took %f secs.' % elapsed_time)

        hdf5_file.close()
        logging.info('Successfully saved HDF5 file %s' % filename)
        sess.close()

def setSimilarity(mtx):
    # mtx is the similarity confusion matrix for all similarities across all vectors in S_a and S_u
    # np.shape(mtx) should be in the form of [#S_a, #S_u] (for toy case, shape=[S_a_i, 256))
    F = np.sum(mtx.max(axis=0))
    return F
def setDistance(mtx):
    '''Function computes the Big F of Set_0 (axis 0) and Set_1 (axis 1) where the mtx is the distance confusion matrix.'''
    F = np.sum(np.min(mtx, axis=0))
    return F

def max_set_cover(distance_mtx, inds_most_certain, return_set_size, verbose=True):
    '''Function computes the max-set cover for a given distance confusion matrix, indices of uncertain samples, and number of representative samples to be picked.'''
    nsamples = distance_mtx.shape[0]
    nlabels = inds_most_certain.shape[0]
    K = inds_most_certain.shape[1]
    max_set = []
    set_dist = []
    if return_set_size > nsamples:
        raise AssertionError('More samples than return_set_size is requested.')
    for ind_lbl in range(nlabels):
        t0 = time.time()
        inds_uncertain = inds_most_certain[ind_lbl]
        inds_argsort = np.argsort(inds_uncertain)
        inds_sorted = inds_uncertain[inds_argsort] #note that array will be monotonically increasing index, and not monotonically increasing uncertainty
        S_a = []
        change_of_set_distance = []
        for i in range(return_set_size):
            if len(S_a) == 0:
                m_f = np.sum(a=distance_mtx, axis=1)
                elm = np.argmin(m_f[inds_sorted]) #pick the min distance element among most certain K elements (S_c)
                S_a.append(inds_sorted[elm])
            else:
                mtx_tmp = distance_mtx[S_a,:] #take current S_a, then we'll add the next best element.
                min_dist = np.inf
                min_dist_ind = -1
                for ind in range(K):
                    ind_uncertain = inds_sorted[ind]
                    if ind_uncertain not in S_a:
                        tmp_S_a = np.concatenate((mtx_tmp, np.reshape(distance_mtx[ind_uncertain,:], newshape=(1,-1))), axis=0)
                        curr_dist = setDistance(tmp_S_a)
                        if curr_dist < min_dist:
                            min_dist = curr_dist
                            min_dist_ind = ind_uncertain
                if verbose:
                    logging.info('k_%d: current distance of S_a to S_u is %.3f' % (i, min_dist))
                change_of_set_distance.append(min_dist)
                S_a.append(min_dist_ind)
        set_dist.append(np.asarray(change_of_set_distance))
        max_set.append(np.asarray(S_a))
        if verbose:
            t1 = time.time()
            logging.info('Computation of max_set_cover took %.3fs for label %d' % (t1-t0, ind_lbl))
    return np.asarray(max_set), np.asarray(set_dist)

def MSE(a,b):
    '''Function computes Mean squared error between a and b numpy.ndarrays'''
    return np.mean(np.square(a-b), keepdims=False)

def compute_distance_matrix(filename, distance_metric='cosine_angle', savename=None, verbose=True):
    hdf5_file = h5py.File(filename, "r")
    if distance_metric == 'cosine_angle':
        image_descriptors = hdf5_file['image_descriptors']
        if verbose:
            logging.info('image_descriptors shape: %s' % (str(image_descriptors.shape))) #shape: [#images, #channels]
        image_descriptors_normalized = image_descriptors/np.tile(np.reshape(
            np.linalg.norm(image_descriptors, axis=1), newshape=(-1,1)), reps=(1,image_descriptors.shape[1]))
        cosine_similarity_mtx = np.matmul(a=image_descriptors_normalized, b=image_descriptors_normalized.transpose())
        if savename is not None:
            d = h5py.File(savename, 'w')
            d.create_dataset(name='similarity_mtx', data=cosine_similarity_mtx, dtype=np.float32)
            d.close()
        return cosine_similarity_mtx
    elif distance_metric == 'MSE':
        abstraction_layer_activations = hdf5_file['abstraction_layer_activations']
        act_shape = abstraction_layer_activations.shape
        if verbose:
            logging.info('abstraction_layer_activations shape: %s' % (str(act_shape))) #shape: [#images, Width, height, #channels/filters]

        n_images = act_shape[0]

        current_i = 0
        if savename is not None:
            if os.path.isfile(savename):
                d = h5py.File(savename, 'a')
                data = dict()
                data['dist_mtx'] = d['dist_mtx'][...]
                MSE_distance_mtx = np.asarray(data['dist_mtx'])
                current_i = d['current_i'][()]
            else:
                d = h5py.File(savename, 'w')
                d.create_dataset(name='current_i', data=0, dtype=np.int32)
                d.create_dataset(name='finished', data=False, dtype=np.bool)
                data = dict()
                data['dist_mtx'] = d.create_dataset(name='dist_mtx', shape=[n_images,n_images], dtype=np.float32)
                MSE_distance_mtx = np.inf * np.ones((n_images, n_images), dtype=np.float32)

            logging.info('current_i is %d' % ((d['current_i'][()])))

        if verbose:
            logging.info('MSE_distance_mtx is now initialized.')
        it_counter = 0

        num_ind = 100 #approx 2x 6Gb each op.
        # num_ind = 200  # approx 2x 25Gb each op.
        # num_ind = 300  # approx 2x 55Gb each op.
        it_num = int(np.ceil(float(n_images-current_i) / float(num_ind)))
        curr_i = int(current_i)
        for i in range(it_num):
            i_range = np.arange(curr_i, curr_i+num_ind)
            if i_range[-1] >= n_images:
                i_range = np.arange(curr_i, n_images)
            curr_i += num_ind
            num_elm_i = len(i_range) #not always equal to num_ind
            curr_j = i_range[0]
            for j in range(i, it_num):
                j_range = np.arange(curr_j, curr_j+num_ind)
                if j_range[-1] >= n_images:
                    j_range = np.arange(curr_j, n_images)
                num_elm_j = len(j_range)
                curr_j += num_ind

                logging.info('i,j range: %s, %s' % ([i_range[0], i_range[0]+num_elm_i], [j_range[0],j_range[0]+num_elm_j]))

                ims_i = np.reshape(abstraction_layer_activations[i_range, ...], newshape=(-1, num_elm_i,1)) #shape: [width*height*#channels, 100, 1]
                # ims_i_transposed = ims_i.transpose((0, 2, 1))  # shape[width*height*#channels, 1, 100]
                ims_j = np.reshape(abstraction_layer_activations[j_range, ...], newshape=(-1, num_elm_j,1)) #shape: [width*height*#channels, 100, 1]
                ims_j_transposed = ims_j.transpose((0, 2, 1))  # shape[width*height*#channels, 1, 100]
                if i ==0 and j == 0:
                    logging.info('debug: ims_i shape: %s,\nims_j_transposed shape: %s' % (str(ims_i.shape), str(ims_j_transposed.shape)))
                #now compute MSE:
                mse_tmp = np.mean(np.square(ims_i-ims_j_transposed), axis=0, keepdims=False) #shape: [100,100]
                MSE_distance_mtx[i_range[0]:i_range[0]+num_elm_i,j_range[0]:j_range[0]+num_elm_j] = mse_tmp

                #Write to HDD
                if savename is not None:
                    data_arr = np.asarray(mse_tmp, dtype=np.float32)
                    data['dist_mtx'][i_range[0]:i_range[0]+num_elm_i,j_range[0]:j_range[0]+num_elm_j, ...] = data_arr

                it_counter += 1
                if it_counter % (it_num * it_num /2) == 0:
                    logging.info('%.2f percent of dist_mtx computation is completed.' % (float(it_counter) / float(it_num * it_num) / 2. * 100.))
                logging.info('it_counter: %d. %.3f percent complete.' % (it_counter, float(it_counter) / float(it_num * it_num) / 2. * 100.))
            if savename is not None:
                d['current_i'][...] = curr_i
                logging.info('current_i is: %d' % (d['current_i'][()]))
        logging.info('upper triangle finished. now filling bottom triangle of MSE_dist_mtx')
        for i in range(n_images):
            for j in range(i,n_images):
                if i != j:
                    if np.isnan(MSE_distance_mtx[i,j]):
                        logging.warning('Warning (i,j):(%d,%d) is NaN' % (i,j))
                    MSE_distance_mtx[j,i] = MSE_distance_mtx[i,j]

        if np.any(np.isnan(MSE_distance_mtx)):
            raise AssertionError('MSE_distance_mtx contains NaNs: %s' % (np.where(np.isnan(MSE_distance_mtx))[0]))
        if savename is not None:
            d['finished'][...] = True
            d.close()
        return MSE_distance_mtx

def check_if_lbl_exists(y):
    labels = np.unique(y)
    return np.asarray([[np.any(y[i,...] == lbl) for i in range(y.shape[0])] for lbl in labels])

def compute_vgg19_tensor_distance(images, filename, exp_config, uncertainty_scores, batch_size=20, distance_metric='MSE', verbose=True, **kwargs):

    hdf5_file = h5py.File(filename, "w")
    assert len(images.shape) == 3 # [#images, height, width]

    nlabels = exp_config.nlabels

    if batch_size % 2 != 0.0:
        raise AssertionError('Batch size needs to be an even number')
    a0_inds = np.arange(batch_size/2, dtype=np.int32)
    a1_inds = np.arange(batch_size/2, batch_size, dtype=np.int32)
    a0v, a1v = np.meshgrid(a0_inds, a1_inds, indexing='ij')
    a0v_flat = np.reshape(a0v, (-1,1))
    a1v_flat = np.reshape(a1v, (-1,1))
    if len(images) < batch_size:
        batch_size = len(images)
    images_shp = images.shape
    num_samples = images_shp[0]

    data_gen_train = utils.utils.pair_iterator_hdf5_samples(images, batch_size=batch_size)

    dist_mtx = np.inf*np.ones((num_samples,num_samples))

    hdf5_file.create_dataset(name='nlabels', data=nlabels, dtype=np.int32)

    total_time = 0
    start_time = time.time()

    #layers of interest:
    # l_names = ['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2']

    g = tf.Graph()
    with g.as_default():
        dataset = tf.data.Dataset.from_generator(lambda: data_gen_train, output_types=[tf.float32, tf.int32])
        dataset = dataset.repeat(count=1)
        dataset = dataset.batch(1, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=batch_size)

        iterator = dataset.make_initializable_iterator()
        batch_op = iterator.get_next()
        x_batch_pl = batch_op[0]
        batch_inds_op = batch_op[1]

        image_tensor_shape = [batch_size] + list(exp_config.image_size) + [1]
        x_batch_pl = tf.tile(tf.reshape(x_batch_pl, shape=image_tensor_shape), multiples=[1,1,1,3]) #make pseudo RGB

        init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
        config.allow_soft_placement = True  # If a operation is not define it the default device, let it execute in another.
        config.gpu_options.per_process_gpu_memory_fraction = 1.0
        sess = tf.Session(config=config)

        sess.run(init)

        data_min_max = [np.min(images), np.max(images)]

        ################################################################################
        #### Please download vgg19 architecture and pretrained weights yourself.
        #### You can use https://github.com/machrisaa/tensorflow-vgg
        #### Note that you need to remove the zero-mean ops from the downloaded code.
        raise AssertionError('Please download vgg19 architecture and pretrained weights yourself.')
        import external.vgg19
        path_vgg19 = './path/to/vgg19/weights/vgg19.npy'
        vgg = external.vgg19.Vgg19(vgg19_npy_path=path_vgg19, dataset_min_max_int=data_min_max)
        norm_min_max = [0,6000] #adapt for your dataset.
        x_batch_mod = (x_batch_pl-norm_min_max[0])/(norm_min_max[1]-norm_min_max[0]) * 255.0
        x_batch_mod = tf.image.resize_images(x_batch_mod, size=[224,224], align_corners=True)
        vgg.build(rgb=x_batch_mod)
        ################################################################################
        act_list = [vgg.conv1_2, vgg.conv2_2, vgg.conv3_2, vgg.conv4_2, vgg.conv5_2]
        if verbose:
            logging.info('act_list tensor shapes: %s' % (str([i.get_shape().as_list() for i in act_list])))

        rm_axes = (1,2,3)
        if distance_metric == 'MSE': #np.mean(np.square(a-b)
            dist_ij_op = tf.reduce_mean(tf.squared_difference(x=tf.gather_nd(act_list[0], a0v_flat), y=tf.gather_nd(act_list[0], a1v_flat)), axis=rm_axes) #shape: [batch_size/2,batch_size/2]
            logging.info('first dist_ij_op shape: %s' % (str(dist_ij_op.get_shape().as_list())))
            for i in range(1,len(act_list)):
                dist_ij_op = dist_ij_op + tf.reduce_mean(tf.squared_difference(x=tf.gather_nd(act_list[i], a0v_flat), y=tf.gather_nd(act_list[i], a1v_flat)), axis=rm_axes)
            if verbose:
                logging.info('ops for MSE computation are defined.')
        elif distance_metric == 'MAE': #np.mean(np.abs(a-b)
            dist_ij_op = tf.reduce_mean(tf.abs(x=tf.gather_nd(act_list[0], a0v_flat), y=tf.gather_nd(act_list[0], a1v_flat)), axis=rm_axes)
            for i in range(1, len(act_list)):
                dist_ij_op = dist_ij_op + tf.reduce_mean(tf.abs(x=tf.gather_nd(act_list[i], a0v_flat), y=tf.gather_nd(act_list[i], a1v_flat)), axis=rm_axes)
            if verbose:
                logging.info('ops for MAE computation are defined.')
        if verbose:
            logging.info('dist_ij_op shape: %s' % (str(dist_ij_op.get_shape().as_list())))


        hdf5_file.create_dataset(name='uncertainty_scores', data=uncertainty_scores, dtype=np.float32)



        # num_total_iterations = int(np.ceil(num_samples / batch_size))
        tmp_val = float(np.ceil(float(num_samples)/float(batch_size/2.)))
        num_total_iterations = int(tmp_val * (tmp_val+1.)/2.) #upper triangle

        for ind in range(num_total_iterations):
            dist_ij, batch_inds = sess.run([dist_ij_op, batch_inds_op])

            dist_mtx[batch_inds[0]:batch_inds[1], batch_inds[2]:batch_inds[3]] = np.reshape(dist_ij, newshape=a0v.shape)

            if verbose and ind % 100 == 0:
                tstmp = time.time()
                est = (tstmp - start_time)*(num_total_iterations/(ind+1) - 1)
                logging.info('\nProcessing minibatch %i out of %i (%i x %i)\ninds: %s.\nEstimated time left: %.3fhours OR %.3fmins OR %.1fs'
                             % (ind + 1, num_total_iterations, tmp_val, (tmp_val + 1.) / 2., str(batch_inds),
                                est/60/60, est/60, est))

        tstmp = time.time()
        est = (tstmp - start_time) * (num_total_iterations / (ind + 1) - 1)
        logging.info('\nProcessed minibatch %i out of %i (%i x %i)\ninds: %s.\nEstimated time left: %.3fhours OR %.3fmins OR %.1fs'
            % (ind + 1, num_total_iterations, tmp_val, (tmp_val + 1.) / 2., str(batch_inds),
               est / 60 / 60, est / 60, est))
    for i in range(num_samples):
        for j in range(i, num_samples):
            if i != j:
                if np.isnan(dist_mtx[i, j]):
                    logging.warning('Warning (i,j):(%d,%d) is NaN' % (i, j))
                dist_mtx[j, i] = dist_mtx[i, j]

    hdf5_file.create_dataset(name='dist_mtx', data=dist_mtx, dtype=np.float32)
    hdf5_file.close()
    logging.info('Successfully saved HDF5 file %s' % filename)
    total_time = time.time() - start_time
    logging.info('It took %.3fmins, OR %.3fh to compute dist_mtx' % (total_time/60., total_time/60./60.))
    sess.close()
    return dist_mtx

def compute_image_descriptors(images, filename, exp_config, checkpoint_path, uncertainty_scores, verbose=True, **kwargs):
    hdf5_file = h5py.File(filename, "w")
    if len(images.shape) == 3:  # single-channel input
        num_input_channels = 1
    elif len(images.shape) == 4:  # multi-channel input
        num_input_channels = images.shape[3]
    nlabels = exp_config.nlabels
    batch_size = exp_config.batch_size
    if len(images) < batch_size:
        batch_size = len(images)
    images_shp = images.shape

    hdf5_file.create_dataset(name='nlabels', data=nlabels, dtype=np.int32)

    data_dict = {'images': images}
    generator = utils.image.DataGenerator(data=data_dict, shuffle=False)

    total_time = 0
    start_time = time.time()

    g = tf.Graph()
    with g.as_default():

        x_shape = list(exp_config.image_size)
        x_dtype = tf.float32

        dataset = tf.data.Dataset.from_generator(lambda: generator, output_shapes={'images': x_shape},
                                                 output_types={'images': x_dtype})
        dataset = dataset.repeat(count=1)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=batch_size)

        # set iterator
        iterator = dataset.make_initializable_iterator()
        batch_op = iterator.get_next()
        if num_input_channels == 1:
            x_batch_pl = tf.expand_dims(batch_op['images'], axis=-1)
        else:
            x_batch_pl = batch_op['images']

        training_pl = tf.constant(False, dtype=tf.bool, shape=[], name='is_training_flag')
        # Define inference graph operation
        model_obj = exp_config.model_class(images=x_batch_pl, nlabels=nlabels,
                                           num_filters_first_layer=exp_config.num_filters_first_layer,
                                           training=training_pl, architecture_mods='incremental')
        act_abstraction_layer = model_obj.abstraction_layer
        image_descriptor_op = tf.reduce_mean(act_abstraction_layer, axis=(1,2), keepdims=False) # shape: [batch_size, #channels]

        saver = tf.train.Saver(max_to_keep=1)

        init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
        config.allow_soft_placement = True  # If a operation is not define it the default device, let it execute in another.
        config.gpu_options.per_process_gpu_memory_fraction = 1.0
        sess = tf.Session(config=config)

        sess.run(init)

        sess.run(iterator.initializer)

        # Restore model weights
        saver.restore(sess, checkpoint_path)

        hdf5_file.create_dataset(name='uncertainty_scores', data=uncertainty_scores, dtype=np.float32)

        num_samples = images_shp[0]


        num_total_iterations = int(np.ceil(num_samples / batch_size))

        data = dict()
        img_descriptors_shape = [num_samples] + image_descriptor_op.get_shape().as_list()[1:]
        if verbose:
            logging.info('img_descriptors_shape: %s' % (str(img_descriptors_shape)))
        data['image_descriptors'] = hdf5_file.create_dataset('image_descriptors', shape=img_descriptors_shape,
                                                              dtype=np.float32)

        train_index_counter = 0  # this is just a security measure to make sure generator indices match original dataset indices

        for ind in range(num_total_iterations):
            if verbose:
                logging.info('Processing minibatch %i out of %i.' % (ind + 1, num_total_iterations))

            train_index_counter = train_index_counter + batch_size

            x_batch, image_descriptors = sess.run([x_batch_pl, image_descriptor_op])

            counter_from = train_index_counter - batch_size
            counter_to = train_index_counter

            external.utils.write_range_to_hdf5(hdf5_data=data, list_name='image_descriptors',
                                               data_list=image_descriptors,
                                               dtype=np.float32, counter_from=counter_from, counter_to=counter_to)

            if ind != 0 and ind % 10 == 0:
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                start_time = time.time()
                logging.info('Evaluation of 10 minibatches took %f secs.' % elapsed_time)

        hdf5_file.close()
        logging.info('Successfully saved HDF5 file %s' % filename)
        sess.close()

def compute_uncertainty(images, filename, exp_config, checkpoint_path, method='mcdropout', dropout_rate=0.5, **kwargs):
    if method not in ['mcdropout', 'BALD']:
        raise AssertionError('Method needs to be mcdropout or BALD for the time being as others are not implemented')

    num_mcdropout = kwargs.get('num_mcdropout', None)
    if num_mcdropout == None:
        raise AssertionError('Unknown num_mcdropout.')

    hdf5_file = h5py.File(filename, "w")
    hdf5_file.create_dataset(name='sample_indices', data=exp_config.inds_train, dtype=np.int32)

    if len(images.shape) == 3:  # single-channel input
        num_input_channels = 1
    elif len(images.shape) == 4:  # multi-channel input
        num_input_channels = images.shape[3]
    nlabels = exp_config.nlabels
    batch_size = exp_config.batch_size
    if len(images) < batch_size:
        batch_size = len(images)
    images_shp = images.shape

    data_dict = {'images': images}
    generator = utils.image.DataGenerator(data=data_dict, shuffle=False)

    total_time = 0
    start_time = time.time()

    g = tf.Graph()
    with g.as_default():
        image_tensor_shape = [batch_size] + list(exp_config.image_size) + [num_input_channels]

        x_shape = list(exp_config.image_size)
        x_dtype = tf.float32

        dataset = tf.data.Dataset.from_generator(lambda: generator, output_shapes={'images':x_shape}, output_types={'images':x_dtype})
        dataset = dataset.repeat(count=1)
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=batch_size)

        iterator = dataset.make_initializable_iterator()
        batch_op = iterator.get_next()

        if num_input_channels == 1:
            x_batch_next = tf.expand_dims(batch_op['images'], axis=-1)
        else:
            x_batch_next = batch_op['images']

        # Create placeholders for all necessary data input to TF Graph
        x_batch_pl = tf.placeholder(tf.float32, shape=[None]+image_tensor_shape[1:], name='images')
        training_pl = tf.constant(False, dtype=tf.bool, shape=[], name='is_training_flag')
        dropout_rate_pl = tf.constant(dropout_rate, dtype=tf.float32, shape=[], name='dropout_rate')
        # Define inference graph operation
        model_obj = exp_config.model_class(images=x_batch_pl, nlabels=nlabels,
                                           num_filters_first_layer=exp_config.num_filters_first_layer,
                                           training=training_pl, dropout_rate=dropout_rate_pl,
                                           architecture_mods='incremental0')
        logits = model_obj.new_head(numClasses=exp_config.nlabels, head_name='Head_0')
        model_obj.model = tf.keras.Model(inputs=model_obj.krs_inputs, outputs=logits)
        sm_logits = tf.nn.softmax(logits)

        saver = tf.train.Saver(max_to_keep=1)

        init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
        config.allow_soft_placement = True  # If an operation is not define in the default device, let it execute in another.
        config.gpu_options.per_process_gpu_memory_fraction = 1.0
        sess = tf.Session(config=config)

        sess.run(init)

        sess.run(iterator.initializer)

        # Restore model weights
        saver.restore(sess, checkpoint_path)

        num_total_iterations = int(np.ceil(images_shp[0] / batch_size))

        data = dict()
        data['images'] = hdf5_file.create_dataset('images', list(images_shp), dtype=np.float32)
        if method == 'mcdropout':
            uncertainty_images_shp = list(images_shp[:3]) +[nlabels]
        elif method == 'BALD':
            uncertainty_images_shp = list(images_shp[:3])
        data['uncertainty_images'] = hdf5_file.create_dataset('uncertainty_images', uncertainty_images_shp, dtype=np.float32)
        data['uncertainty_scores'] = hdf5_file.create_dataset('uncertainty_scores', [images_shp[0]] + [nlabels], dtype=np.float32)
        data['averaged_uncertainty_scores'] = hdf5_file.create_dataset('averaged_uncertainty_scores', [images_shp[0]],dtype=np.float32)

        train_index_counter = 0

        for ind in range(num_total_iterations):
            logging.info('Processing minibatch %i out of %i.' % (ind + 1, num_total_iterations))

            train_index_counter = train_index_counter + batch_size

            x_batch = sess.run(x_batch_next)
            logit_preds = []
            for i in range(num_mcdropout):
                # curr_dr = dropout_rate
                logit_preds.append(sess.run(sm_logits, feed_dict={x_batch_pl: x_batch}))
                # logit_preds.append(sess.run(sm_logits))

            logit_preds = np.asarray(logit_preds) #shape: [#mcdropout, batch_size, width, height, nlabels]
            if method == 'mcdropout':
                uncertainty_images = np.var(logit_preds, axis=0, keepdims=False) #shape: [Batch_size, width, height, nlabels ]
                uncertainty_scores = np.mean(uncertainty_images, axis=(1,2), keepdims=False) #shape: [batch_size, nlabels]
                uncertainty_scores_lbl_avg = np.mean(uncertainty_scores, axis=1, keepdims=False) #shape: [batch_size]
            elif method == 'BALD':
                pred_MAP = np.mean(logit_preds, axis=0, keepdims=False)
                uncertainty_images = -1*np.sum(pred_MAP*np.log(pred_MAP), axis=-1, keepdims=False) + np.mean(np.sum(logit_preds*np.log(logit_preds), axis=-1, keepdims=False), axis=0, keepdims=False) #shape: [batch_size, width, height]
                uncertainty_scores_lbl_avg = np.mean(uncertainty_images, axis=(1,2), keepdims=False) #shape: [batch_size]
                uncertainty_scores = np.tile(np.expand_dims(uncertainty_scores_lbl_avg, axis=-1), reps=(1, nlabels)) #shape: [batch_size, nlabels]


            counter_from = train_index_counter - batch_size
            counter_to = train_index_counter

            external.utils.write_range_to_hdf5(hdf5_data=data, list_name='images', data_list=np.squeeze(x_batch), dtype=np.float32,
                                counter_from=counter_from, counter_to=counter_to)
            external.utils.write_range_to_hdf5(hdf5_data=data, list_name='uncertainty_images', data_list=uncertainty_images,
                                               dtype=np.float32, counter_from=counter_from, counter_to=counter_to)
            external.utils.write_range_to_hdf5(hdf5_data=data, list_name='uncertainty_scores',
                                               data_list=uncertainty_scores, dtype=np.float32,
                                               counter_from=counter_from, counter_to=counter_to)
            external.utils.write_range_to_hdf5(hdf5_data=data, list_name='averaged_uncertainty_scores',
                                               data_list=uncertainty_scores_lbl_avg, dtype=np.float32,
                                               counter_from=counter_from, counter_to=counter_to)

            if ind != 0 and ind % 10 == 0:
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                start_time = time.time()
                logging.info('Evaluation of 10 minibatches took %f secs.' % elapsed_time)

        hdf5_file.close()
        logging.info('Successfully saved HDF5 file %s' % filename)
        sess.close()