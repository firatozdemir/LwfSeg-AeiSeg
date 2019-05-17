# Authors:
# Firat Ozdemir (fozdemir@gmail.com), 2019, ETH Zurich

import tensorflow as tf
import utils.image
import utils.utils
import numpy as np
import h5py
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class ExperimentConfig:
    """ Simple class to manage seeds for experimental repeatability."""
    def __init__(self):
        self.label_name='SKI10_1femur_2tibia'
        self.num_filters_first_layer = 32
        self.padding_type = 'SAME'
        self.dropout_rate = 0.5
        self.modality_str = 'MR'


        # Source files to be copied at log-dir for future checking
        self.source_to_copy = ['experiment_configurations/unet2D_SKI10_resize_1femur_2tibia_IJCARS2019.py', 'inc_train.py', 'utils', 'wrappers', 'architectures', 'external']

        self._DEBUG = True

        # Model settings
        self.model_handle = None
        self.model_class = None

        # Data settings
        self.dataset_parent_folder = 'path/to/preprocessed/data/with/BFC/correction/and/resized/concatenated/2d/slices'

        self.use_Patches = True
        self.data_mode = '2D'  # 2D or 3D
        self.available_sizes = [(128,128), (224,224), (228,228)]
        self.image_size = self.available_sizes[1]
        self.target_resolution = ['various']
        self.nlabels = 2
        self.foreground_lbl_range = [1, None] #For evaluation ## Deprecated
        self.foreground_lbls = [1,2] #Note the set is [start, end), so end index is not included.
        self.dataset = 'data_2D_size_{!s}_{!s}_res_various.hdf5'.format(
            str(self.image_size[0]), str(self.image_size[1]))
        self.dataset_GT = 'data_2D_size_original_res_original.hdf5'


        # Training settings
        self.batch_size = 4
        self.learning_rate = 1.e-3
        self.optimizer_handle = tf.train.AdamOptimizer
        self.loss_type = 'weighted_crossentropy'
        self.ratio_train_valid_test_options = [[0.35, 0.35, 0.05, 0.25], [0.69, 0.01, 0.05, 0.25],
                                               [0.60, 0.10, 0.05, 0.25], [0.67, 0.03, 0.05, 0.25]]
        self.case_ind = 0 #0: case1, 1: case2,  2: case3,  3: case4
        self.ratio_train_valid_test = self.ratio_train_valid_test_options[self.case_ind]
        self.initial_seed = 1991

        # Augmentation settings
        channel_dim_exist = False
        # flip_up_down_config = {'flip_chance': 0.5, 'channel_dim_exist': channel_dim_exist}
        flip_left_right_config = {'flip_chance': 0.5, 'channel_dim_exist': channel_dim_exist}
        # rotate_config = {'max_angle': 30, 'unidirectional': False, 'channel_dim_exist': channel_dim_exist}
        resize_config = {'max_ratio': 1.5, 'only_zoom': False, 'resize_uniform': True,
                         'channel_dim_exist': channel_dim_exist}
        self.data_augment_config = {
            utils.image._resize: resize_config
            # ,utils.image._flip_up_down: flip_up_down_config
            , utils.image._flip_left_right: flip_left_right_config
            # ,utils.image._rotate: rotate_config
        }

        self.experiment_name = None# exp_name()+'Case'+str(case_ind+1)

        self.seed_keeper = utils.utils.SeedKeeper(initial_seed=self.initial_seed)
        self.compute_sample_indices()


        #Test settings: Dataset dependent parameters for test script (i.e., UNet_tf_test.py)

        self.max_epochs = 1000
        self.max_steps = 20000

        self.train_eval_frequency = 10
        self.val_eval_frequency = 10

        # Creation of exemplar dataset
        self.MCUncertainty_count = 29
        self.K_most_certain = 1000
        self.k_representative = 100
        # Active Learning Iteration 1 Dataset
        self.Iteration1 = Iteration1(self)

    def set_initial_seed(self, seedID):
        if seedID == '1':
            if self.initial_seed == 1991:
                return
            else:
                logging.info('Changing seed ID to 1 in configuration from %d to 1991' % (self.initial_seed))
                self.initial_seed = 1991
        elif seedID == '2':
            if self.initial_seed == 1881:
                return
            else:
                logging.info('Changing seed ID to 2 in configuration from %d to 1881' % (self.initial_seed))
                self.initial_seed = 1881
        elif seedID == '3':
            if self.initial_seed == 1938:
                return
            else:
                logging.info('Changing seed ID to 3 in configuration from %d to 1938' % (self.initial_seed))
                self.initial_seed = 1938
        elif seedID == '4':
            if self.initial_seed == 905:
                return
            else:
                logging.info('Changing seed ID to 4 in configuration from %d to 905' % (self.initial_seed))
                self.initial_seed = 905
        elif seedID == '5':
            if self.initial_seed == 42:
                return
            else:
                logging.info('Changing seed ID to 5 in configuration from %d to 42' % (self.initial_seed))
                self.initial_seed = 42
        else:
            raise AssertionError('Unknown experiment seed ID entered.')

        self.seed_keeper = utils.utils.SeedKeeper(initial_seed=self.initial_seed)
        self.compute_sample_indices()
        self.Iteration1.set_inds(self)
    def set_case(self, caseID):
        if caseID == '1':
            if self.case_ind != 0:
                logging.info('Changing Case ID in configuration to 0 from %d' % (self.case_ind))
                self.case_ind = 0
                self.ratio_train_valid_test = self.ratio_train_valid_test_options[self.case_ind]
                self.seed_keeper = utils.utils.SeedKeeper(initial_seed=self.initial_seed)
                self.compute_sample_indices()
                self.Iteration1.set_inds(self)
        elif caseID == '2':
            if self.case_ind != 1:
                logging.info('Changing Case ID in configuration to 1 from %d' % (self.case_ind))
                self.case_ind = 1
                self.ratio_train_valid_test = self.ratio_train_valid_test_options[self.case_ind]
                self.seed_keeper = utils.utils.SeedKeeper(initial_seed=self.initial_seed)
                self.compute_sample_indices()
                self.Iteration1.set_inds(self)
        elif caseID == '3':
            if self.case_ind != 2:
                logging.info('Changing Case ID in configuration to 2 from %d' % (self.case_ind))
                self.case_ind = 2
                self.ratio_train_valid_test = self.ratio_train_valid_test_options[self.case_ind]
                self.seed_keeper = utils.utils.SeedKeeper(initial_seed=self.initial_seed)
                self.compute_sample_indices()
                self.Iteration1.set_inds(self)
        elif caseID == '4':
            if self.case_ind != 3:
                logging.info('Changing Case ID in configuration to 3 from %d' % (self.case_ind))
                self.case_ind = 3
                self.ratio_train_valid_test = self.ratio_train_valid_test_options[self.case_ind]
                self.seed_keeper = utils.utils.SeedKeeper(initial_seed=self.initial_seed)
                self.compute_sample_indices()
                self.Iteration1.set_inds(self)
        else:
            raise AssertionError('Unknown experiment scenario ID entered.')
    def set_inds_to_incremental_in_init(self):
        self.inds_train = self.Iteration1.inds_train

    def compute_sample_indices(self):
        fname = os.path.join(self.dataset_parent_folder, self.dataset)
        data = h5py.File(fname, 'r')
        pID_list = list(data['PatientID'])
        ids_patients = np.unique(pID_list)
        num_total_data = len(ids_patients)
        ind_range = list(range(num_total_data))
        self.seed_keeper.prng.shuffle(ind_range)
        ratioTVT = self.ratio_train_valid_test
        indStartTVT = [int(np.floor(num_total_data * np.sum(ratioTVT[:i]))) for i in range(len(ratioTVT) + 1)]
        inds_train = [np.where(pID_list == ids_patients[i1])[0] for i1 in ind_range[indStartTVT[0]:indStartTVT[1]]]
        self.inds_train = np.sort([i for subl in inds_train for i in subl])
        inds_validation = [np.where(pID_list == ids_patients[i1])[0] for i1 in ind_range[indStartTVT[2]:indStartTVT[3]]]
        self.inds_validation = np.sort([i for subl in inds_validation for i in subl])
        inds_test = [np.where(pID_list == ids_patients[i1])[0] for i1 in ind_range[indStartTVT[3]:]]
        self.inds_test = np.sort([i for subl in inds_test for i in subl])

        inc_inds_train = [np.where(pID_list == ids_patients[i1])[0] for i1 in ind_range[indStartTVT[1]:indStartTVT[2]]]
        self.inc_inds_train = np.sort([i for subl in inc_inds_train for i in subl])
        data.close()

    def modify_GT_Head_0(self, gt):  # only keep femur

        gtn = np.copy(gt)
        gtn[gtn > 1] = 0
        return gtn


    def modify_GT_Head_1(self, gt):  # only keep tibia
        gtn = np.copy(gt)
        gtn[gtn != 3] = 0
        gtn[gtn == 3] = 1
        return gtn


class Iteration1:
    def __init__(self, parent_class):
        self.dropout_rate = parent_class.dropout_rate

        # Data settings
        self.iteration_name = 'increment1'
        self.dataset_exempler = 'exemplar_SKI10Femur'
        self.dataset = parent_class.dataset
        self.dataset_parent_folder = parent_class.dataset_parent_folder
        self.use_Patches = True
        self.data_mode = '2D'  # 2D or 3D
        self.image_size = parent_class.image_size
        self.target_resolution = parent_class.target_resolution
        self.nlabels = 2 #use for softmax cross entropy
        self.nFGlabels = 1 #Use for class independent sigmoid
        self.foreground_lbl_range = [1, None] #Deprecated
        self.foreground_lbls = [1,2] #Note the set is [start, end), so end index is not included.
        self.inds_train = parent_class.inc_inds_train
        self.inds_validation = parent_class.inds_validation
        self.inds_test = parent_class.inds_test

        # Training settings
        self.batch_size = parent_class.batch_size
        self.learning_rate = parent_class.learning_rate
        self.optimizer_handle = tf.train.AdamOptimizer
        self.loss_type = 'weighted_crossentropy'

        # Augmentation settings
        self.data_augment_config = parent_class.data_augment_config
    def set_inds(self, parent_class):
        self.inds_train = parent_class.inc_inds_train
        self.inds_validation = parent_class.inds_validation
        self.inds_test = parent_class.inds_test
