function Params = config_bt(base_path)
% [Params] = config_bt
%   specifies parameters for beat tracking algorithm
% ----------------------------------------------------------------------
% INPUT Parameter:
%   none
%
% OUTPUT Parameter:
%   Params            : structure array with beat tracking parameters
%
% 06.09.2012 by Florian Krebs
% ----------------------------------------------------------------------

% system name
Params.system = 'BeatTracker';

% Path settings
if exist('base_path', 'var')
    Params.base_path = base_path;
else
    Params.base_path = '~/diss/src/matlab/beat_tracking/bayes_beat';
end
Params.data_path = fullfile(Params.base_path, 'data');
Params.results_path = fullfile(Params.base_path, 'results');
Params.temp_path = fullfile(Params.base_path, 'temp');

% SIMULATION PARAMETERS:
% ======================

% If n_depends_on_r=true, then use different tempo limits for each rhythm
% state
Params.n_depends_on_r = 1;
% If patternGiven=true, then take the pattern labels as given
Params.patternGiven = 0;
% n_folds_for_cross_validation
%   0) use train and test set as described below
%   1) use leave-one-out splitting (train and test are the same)
%   k) use k-fold cross-validation (train and test are the same)
Params.n_folds_for_cross_validation = 0;
% If save_inference_data=true, then save complete posterior probability to
% file. This is useful for visualisations.
Params.save_inference_data = 0;
% If reorganize_bars_into_cluster=true, then reorganise features into
% patterns as given by the cluster_assignment_file. Otherwise, Data.extract_feats_per_file_pattern_barPos_dim 
%is loaded from file.
Params.reorganize_bars_into_cluster = 0; % reorganize in Data.extract_feats_per_file_pattern_barPos_dim
% Inference and model settings {'HMM_viterbi', 'HMM_forward', 'PF',
% 'PF_viterbi'}
Params.inferenceMethod = 'HMM_viterbi';
% Number of iterations of Viterbi training (currently only for HMMs)
Params.viterbi_learning_iterations = 0;
% Filename of pre-stored model to load
% Params.model_fln = fullfile(Params.temp_path, 'last_model.mat');
% Params.model_fln = '~/diss/src/matlab/beat_tracking/bayes_beat/data/big_hmm_carnatic_beats.mat';
% Save extracted feature to a folder called "beat_activations" relative to
% the audio folder
Params.save_features_to_file = 1;
Params.use_mex_viterbi = 0;

% SYSTEM PARAMETERS:
% ==================

% State space size
% ----------------

% Maximum position state (used for the meter with the longest duration)
Params.M = 640;
% Maximum tempo state 
Params.N = 23;
% Number of rhythmic pattern states
Params.R = 1;
% Number of position grid points per whole note. This is important for the
% observation model, as parameters are tied within this grid.
Params.whole_note_div = 64; 
% Length of rhythmic patterns {beat', 'bar'}
Params.pattern_size = 'beat'; % 'beat' or 'bar'
% Audio frame length [sec]
Params.frame_length = 0.01;
% Model initial distribution over tempo states by mixture of init_n_gauss
% Gaussians.
Params.init_n_gauss = 0;
% Use one state to detect silence
Params.use_silence_state = 0;
% Probability of entering the silence state
Params.p2s = 0.00001; % 0.00001
% Probability of leaving the silence state
Params.pfs = 0.001; % 0.001
% File from which the silence observation model params are learned
Params.silence_lab = '~/diss/data/beats/lab_files/robo_silence.lab';
% In online mode (forward path), the best state is chosen among a set of
% possible successor state. This set contains position states within a window
% of +/- max_shift frames
Params.online.max_shift = 4;
% In online mode, we reset the best state sequence to the global best state
% each update_interval
Params.online.update_interval = 1000;
% To avoid overfitting and prevent the obs_lik to become zero, we set a
% floor
Params.online.obs_lik_floor = 1e-7;
% Probability of rhythmic pattern change
Params.pr = 0;
Params.correct_beats = 2;
% Set tempo limits (same for all rhythmic patterns). If no ranges are given, they are learned from data.
Params.min_tempo = 60;
Params.max_tempo = 215;

% HMM parameters
% --------------

% Probability of tempo acceleration (and deceleration)
Params.pn = 0.001;  
% Settings for Viterbi learning: tempo_tying
%   0) p_n tied across position states (different p_n for each n)
%   1) Global p_n for all changes (only one p_n)
%   2) Separate p_n for tempo increase and decrease (two different p_n)
Params.tempo_tying = 1; 


% PF parameters
% -------------

% Number of particles
Params.nParticles = 1000;
% Standard deviation of tempo transition. Note that the tempo n is normalised
% by dividing by M, so the actual sigma is sigmaN * M.
Params.sigmaN = 0.0001; 
% If the effective sample size is below ratio_Neff * nParticles, resampling is performed.
Params.ratio_Neff = 0.1;
Params.res_int = 30;
% Type of resampling scheme to be used:
%   0) Standard SISR (systematic resampling)
%   1) APF
%   2) Mixture PF using k-means clustering (MPF)
%   3) Auxiliary mixture particle filter (AMPF)
Params.resampling_scheme = 0;

Params.do_viterbi_filtering = 0;

% APF parameters
% ..............
% Warping function of weights for APF and AMPF
Params.warp_fun = '@(x)x.^(1/4)';

% Mixture PF parameters
% .....................
% Factors to adjust distance function for k-means [l_m, l_n, l_r]
Params.state_distance_coefficients = [30, 1, 100];
% If distance < cluster_merging_thr: merge clusters
Params.cluster_merging_thr = 20; 
% If spread > cluster_splitting_thr: split clusters
Params.cluster_splitting_thr = 30; 
% If number of clusters > n_max_clusters, kill cluster with lowest weight
Params.n_max_clusters = 100;
% Number of cluster to start with
Params.n_initial_clusters = 32;

% Observation model
% -----------------

% Distribution type {invGauss, fixed, gamma, histogram, multivariateHistogram,
% bivariateGauss, mixOfGauss, MOG, MOG3, ...}
Params.observationModelType = 'RNN';
% Features (extension) to be used
% Params.feat_type{1} = 'sprflx2d0';
% Params.feat_type{2} = 'sprflx2d1';
%Params.feat_type{1} = 'sprflx';
%   Params.feat_type{1} = 'lo230_superflux.mvavg';
%   Params.feat_type{2} = 'hi250_superflux.mvavg';
Params.feat_type{1} = 'rnn_orig';
% Params.feat_type{1} = 'rnn_hainsworth';
% Params.feat_type{1} = 'sprflx-online';
% Feature dimension
Params.featureDim = length(Params.feat_type);

% DATA:
% =====

% Train data
% ----------

% Train dataset
Params.train_set = 'ballroom_beatles_boeck_rwc_2_3_4';
% Path to lab file
Params.trainLab =  ['~/diss/data/beats/lab_files/', Params.train_set, '.lab'];
% Path to file where pattern transitions are stored
%   Params.cluster_transitions_fln = fullfile(Params.data_path, ['cluster_transitions-', ...
%       Params.train_set, '-', num2str(Params.featureDim), 'd-', num2str(Params.R), '.txt']);
% Path to file where cluster to bar assignments are stored
Params.clusterIdFln = fullfile(Params.data_path, ['ca-', Params.train_set, '-', num2str(Params.featureDim), 'd-', ...
    num2str(Params.R), 'R-kmeans.mat']);

% Test data
% ----------

% Test dataset
Params.test_set = 'boeck_3_4';
% Path to lab file (.lab) or to test song (.wav)
% Params.testLab = ['~/diss/data/beats/lab_files/', Params.test_set, '.lab'];
% Params.testLab = '~/diss/data/beats/boeck/train12.wav';
Params.testLab = '~/diss/projects/ismir_beats_2014/data/orig/sh_003.beats.txt';
% Params.testLab = '~/diss/projects/ismir_beats_2014/lab_files/hainsworth_orig.lab';

end
