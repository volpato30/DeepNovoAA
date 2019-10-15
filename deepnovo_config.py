# Copyright 2017 Hieu Tran. All Rights Reserved.
#
# DeepNovo is publicly available for non-commercial uses.
# ==============================================================================

"""TODO(nh2tran): docstring."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


# ==============================================================================
# FLAGS (options) for this app
# ==============================================================================


tf.app.flags.DEFINE_string("train_dir", # flag_name
                           "train", # default_value
                           "Training directory.") # docstring

tf.app.flags.DEFINE_boolean("reset_step",
                            False, # default_value
                            "Set to true to reset the global step after loading a pretrained model.")

tf.app.flags.DEFINE_integer("direction",
                            2,
                            "Set to 0/1/2 for Forward/Backward/Bi-directional.")

tf.app.flags.DEFINE_boolean("use_intensity",
                            True,
                            "Set to True to use intensity-model.")

tf.app.flags.DEFINE_boolean("shared",
                            False,
                            "Set to True to use shared weights.")

tf.app.flags.DEFINE_boolean("use_lstm",
                            True,
                            "Set to True to use lstm-model.")

tf.app.flags.DEFINE_boolean("lstm_kmer",
                            False,
                            "Set to True to use lstm model on k-mers instead of full sequence.")

tf.app.flags.DEFINE_boolean("knapsack_build",
                            False,
                            "Set to True to build knapsack matrix.")

tf.app.flags.DEFINE_boolean("train",
                            False,
                            "Set to True for training.")

tf.app.flags.DEFINE_boolean("test_true_feeding",
                            False,
                            "Set to True for testing.")

tf.app.flags.DEFINE_boolean("decode",
                            False,
                            "Set to True for decoding.")

tf.app.flags.DEFINE_boolean("beam_search",
                            False,
                            "Set to True for beam search.")

tf.app.flags.DEFINE_integer("beam_size",
                            5,
                            "Number of optimal paths to search during decoding.")


tf.app.flags.DEFINE_boolean("search_denovo",
                            False,
                            "Set to True to do a denovo search.")

tf.app.flags.DEFINE_boolean("test",
                            False,
                            "Set to True to test the prediction accuracy.")

tf.app.flags.DEFINE_boolean("header_seq",
                            True,
                            "Set to False if peptide sequence is not provided.")

tf.app.flags.DEFINE_boolean("decoy",
                            False,
                            "Set to True to search decoy database.")

tf.app.flags.DEFINE_integer("multiprocessor",
                            1,
                            "Use multi processors to read data during training.")


# I/O arguments
tf.app.flags.DEFINE_string("train_spectrum",
                           "train_spectrum",
                           "Spectrum mgf file to train a new model.")
tf.app.flags.DEFINE_string("train_feature",
                           "train_feature",
                           "Feature csv file to train a new model.")
tf.app.flags.DEFINE_string("valid_spectrum",
                           "valid_spectrum",
                           "Spectrum mgf file for validation during training.")
tf.app.flags.DEFINE_string("valid_feature",
                           "valid_feature",
                           "Feature csv file for validation during training.")
tf.app.flags.DEFINE_string("test_spectrum",
                           "test_spectrum",
                           "Spectrum mgf file for testing.")
tf.app.flags.DEFINE_string("test_feature",
                           "test_feature",
                           "Feature csv file for testing.")
tf.app.flags.DEFINE_string("denovo_spectrum",
                           "denovo_spectrum",
                           "Spectrum mgf file to perform de novo sequencing.")
tf.app.flags.DEFINE_string("denovo_feature",
                           "denovo_feature",
                           "Feature csv file to perform de novo sequencing.")
tf.app.flags.DEFINE_string("target_file",
                           "target_file",
                           "Target file to calculate the prediction accuracy.")
tf.app.flags.DEFINE_string("predicted_file",
                           "predicted_file",
                           "Predicted file to calculate the prediction accuracy.")


FLAGS = tf.app.flags.FLAGS


# ==============================================================================
# GLOBAL VARIABLES for VOCABULARY
# ==============================================================================


# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_START_VOCAB = [_PAD, _GO, _EOS]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2

vocab_reverse = ['A',
                 'R',
                 'N',
                 'N(Deamidation)',
                 'D',
                 'C',
                 #'C(Carbamidomethylation)',
                 'E',
                 'Q',
                 'Q(Deamidation)',
                 'G',
                 'H',
                 'I',
                 'L',
                 'K',
                 'M',
                 'M(Oxidation)',
                 'F',
                 'P',
                 'S',
                 'T',
                 'W',
                 'Y',
                 'V',
                ]

vocab_reverse = _START_VOCAB + vocab_reverse
print("vocab_reverse ", vocab_reverse)

vocab = dict([(x, y) for (y, x) in enumerate(vocab_reverse)])
print("vocab ", vocab)

vocab_size = len(vocab_reverse)
print("vocab_size ", vocab_size)


# ==============================================================================
# GLOBAL VARIABLES for THEORETICAL MASS
# ==============================================================================


mass_H = 1.0078
mass_H2O = 18.0106
mass_NH3 = 17.0265
mass_N_terminus = 1.0078
mass_C_terminus = 17.0027
mass_CO = 27.9949

mass_AA = {'_PAD': 0.0,
           '_GO': mass_N_terminus-mass_H,
           '_EOS': mass_C_terminus+mass_H,
           'A': 71.03711, # 0
           'R': 156.10111, # 1
           'N': 114.04293, # 2
           'N(Deamidation)': 115.02695,
           'D': 115.02694, # 3
           'C': 103.00919, # 4
           #'C(Carbamidomethylation)': 161.01919, # C(+58.01) # orbi
           #'C(Carbamidomethylation)': 160.03065, # C(+57.02)
           'E': 129.04259, # 5
           'Q': 128.05858, # 6
           'Q(Deamidation)': 129.0426,
           'G': 57.02146, # 7
           'H': 137.05891, # 8
           'I': 113.08406, # 9
           'L': 113.08406, # 10
           'K': 128.09496, # 11
           'M': 131.04049, # 12
           'M(Oxidation)': 147.0354,
           'F': 147.06841, # 13
           'P': 97.05276, # 14
           'S': 87.03203, # 15
           'T': 101.04768, # 16
           'W': 186.07931, # 17
           'Y': 163.06333, # 18
           'V': 99.06841, # 19
          }

mass_ID = [mass_AA[vocab_reverse[x]] for x in range(vocab_size)]
mass_ID_np = np.array(mass_ID, dtype=np.float32)

mass_AA_min = mass_AA["G"] # 57.02146


# ==============================================================================
# GLOBAL VARIABLES for PRECISION, RESOLUTION, temp-Limits of MASS & LEN
# ==============================================================================


# if change, need to re-compile cython_speedup << NO NEED
# ~ SPECTRUM_RESOLUTION = 10 # bins for 1.0 Da = precision 0.1 Da
# ~ SPECTRUM_RESOLUTION = 20 # bins for 1.0 Da = precision 0.05 Da
# ~ SPECTRUM_RESOLUTION = 40 # bins for 1.0 Da = precision 0.025 Da
SPECTRUM_RESOLUTION = 50 # bins for 1.0 Da = precision 0.02 Da
# ~ SPECTRUM_RESOLUTION = 100 # bins for 1.0 Da = precision 0.01 Da
print("SPECTRUM_RESOLUTION ", SPECTRUM_RESOLUTION)

# if change, need to re-compile cython_speedup << NO NEED
WINDOW_SIZE = 10 # 10 bins
print("WINDOW_SIZE ", WINDOW_SIZE)

# skip peptide mass > MZ_MAX
MZ_MAX = 3000.0
MAX_NUM_PEAK = 1000

KNAPSACK_AA_RESOLUTION = 10000 # 0.0001 Da
mass_AA_min_round = int(round(mass_AA_min * KNAPSACK_AA_RESOLUTION)) # 57.02146
KNAPSACK_MASS_PRECISION_TOLERANCE = 100 # 0.01 Da
num_position = 0

PRECURSOR_MASS_PRECISION_TOLERANCE = 0.01

# during training or test_true_feeding: 
# skip peptide length > MAX_LEN
# assign peptides to buckets of the same length for efficient padding
MAX_LEN = 60 if FLAGS.search_denovo else 30

# ==============================================================================
# HYPER-PARAMETERS of the NEURAL NETWORKS
# ==============================================================================
num_ion = 12
print("num_ion ", num_ion)

weight_decay = 0.0  # no weight decay lead to better result.
print("weight_decay ", weight_decay)

# ~ encoding_cnn_size = 4 * (RESOLUTION//10) # 4 # proportion to RESOLUTION
# ~ encoding_cnn_filter = 4
# ~ print("encoding_cnn_size ", encoding_cnn_size)
# ~ print("encoding_cnn_filter ", encoding_cnn_filter)

embedding_size = 512
print("embedding_size ", embedding_size)

num_lstm_layers = 1
num_units = 64
lstm_hidden_units = 512
print("num_lstm_layers ", num_lstm_layers)
print("num_units ", num_units)

dropout_rate = 0.25

batch_size = 16
num_workers = 6
print("batch_size ", batch_size)

num_epoch = 20

init_lr = 1e-3

max_gradient_norm = 5.0
print("max_gradient_norm ", max_gradient_norm)


focal_loss = True

batch_size = 32
print("batch_size ", batch_size)


steps_per_validation = 300  # 100 # 2 # 4 # 200
print("steps_per_validation ", steps_per_validation)

# ==============================================================================
# INPUT/OUTPUT FILES
# ==============================================================================


# pre-built knapsack matrix
knapsack_file = "knapsack.npy"

input_spectrum_file_train = FLAGS.train_spectrum
input_feature_file_train = FLAGS.train_feature
input_spectrum_file_valid = FLAGS.valid_spectrum
input_feature_file_valid = FLAGS.valid_feature
input_spectrum_file_test = FLAGS.test_spectrum
input_feature_file_test = FLAGS.test_feature

# denovo files
# ~ denovo_input_spectrum_file = "data.training/aa.hla.bassani.nature_2016.mel_16.class_1/spectrum.mgf"
# ~ denovo_input_feature_file = "data.training/aa.hla.bassani.nature_2016.mel_16.class_1/feature.csv.mass_corrected"
denovo_input_spectrum_file = FLAGS.denovo_spectrum
denovo_input_feature_file = FLAGS.denovo_feature
denovo_output_file = denovo_input_feature_file + ".deepnovo_denovo"

# test accuracy
predicted_format = "deepnovo"
# ~ target_file = "data.training/aa.hla.bassani.nature_2016.mel_16.class_1/feature.csv.labeled.mass_corrected"
# ~ predicted_file = "data.training/aa.hla.bassani.nature_2016.mel_16.class_1/feature.csv.mass_corrected.deepnovo_denovo.top95.I_to_L.consensus.minlen5"
target_file = FLAGS.target_file
predicted_file = FLAGS.predicted_file
accuracy_file = predicted_file + ".accuracy"
denovo_only_file = predicted_file + ".denovo_only"
scan2fea_file = predicted_file + ".scan2fea"
multifea_file = predicted_file + ".multifea"

# feature file column format
col_feature_id = "spec_group_id"
col_precursor_mz = "m/z"
col_precursor_charge = "z"
col_rt_mean = "rt_mean"
col_raw_sequence = "seq"
col_scan_list = "scans"
col_feature_area = "feature area"

# predicted file column format
pcol_feature_id = 0
pcol_feature_area = 1
pcol_sequence = 2
pcol_score = 3
pcol_position_score = 4
pcol_precursor_mz = 5
pcol_precursor_charge = 6
pcol_protein_id = 7
pcol_scan_list_middle = 8
pcol_scan_list_original = 9
pcol_score_max = 10

distance_scale_factor = 100.
sinusoid_base = 30000.
spectrum_reso = 10
n_position = int(MZ_MAX) * spectrum_reso
