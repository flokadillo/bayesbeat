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
if exist(Params.base_path, 'dir')
    addpath(Params.base_path)
else
    error('Please specify path to bayes_beat class in the config file\n');
end

Params.data_path = fullfile(Params.base_path, 'data');
Params.results_path = fullfile(Params.base_path, 'results');

% SIMULATION PARAMETERS:
% ======================

% If n_depends_on_r=true, then use different tempo limits for each rhythm
% state
Params.n_depends_on_r = 1;
% validation_type
%   'holdout' use train and test set as described below
%   'leave_one_out' use leave-one-out splitting (train and test are the same)
%   'cross_validation' use k-fold cross-validation (train and test are the same)
Params.validation_type = 'cross_validation';
Params.n_folds = 10;
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
% Save extracted feature to a folder called "beat_activations" relative to
% the audio folder
Params.save_features_to_file = 1;
% Load pre-computed features from file
Params.load_features_from_file = 1;
% Use mex implementation of viterbi decoding
Params.use_mex_viterbi = 1;
% Save beat times and corresponding position within a bar (.beats.txt)
Params.save_beats = 1;
% Save only downbeats (.downbeats.txt)
Params.save_downbeats = 0;
% Save median tempo (.bpm.txt)
Params.save_tempo = 0;
% Save rhythm (.rhythm.txt)
Params.save_rhythm = 0;
% Save time_signature (.meter.txt)
Params.save_meter = 0;



% SYSTEM PARAMETERS:
% ==================

% State space size
% ----------------

% Maximum position state (used for the meter with the longest duration)
Params.M = 1600;
% 'Whiteley tm': Maximum tempo state  , '2015 tm': Number of tempo states,
% set to nan if you want to use the maximum number of tempo states possible
Params.N = nan;
% Number of rhythmic pattern states
Params.R = 2;
% Number of position grid points per whole note. This is important for the
% observation model, as parameters are tied within this grid.
Params.whole_note_div = 64; 
% Length of rhythmic patterns {beat', 'bar'}
Params.pattern_size = 'bar'; % 'beat' or 'bar'
% Audio frame length [sec]
Params.frame_length = 0.02;
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
% Correct beat position afterwards by shifting it to a loacl max of the
% onset detection function to correct for the rough discretisation of the
% observation model
Params.correct_beats = 0;
% Learn tempo ranges from data
Params.learn_tempo_ranges = 1;
% Set tempo limits (same for all rhythmic patterns). If 
% learn_tempo_ranges == 1, the tempi learned tempi can be restricted to
% a certain range given by min_tempo and max_tempo [BPM]
Params.min_tempo = 60;
Params.max_tempo = 230;
% When learning tempo ranges, outlier beat intervals can be ignored, e.g.,
% outlier_percentile = 50 uses the median tempo only, = 0 uses all periods
Params.outlier_percentile = 5;

% HMM parameters
% --------------

% Probability of tempo acceleration (and deceleration) in the whiteley
% model
Params.pn = 0.001;  
% squeezing factor for the tempo change distribution
%  (higher values prefer a constant tempo over a tempo
%               change from one beat to the next one)
Params.alpha = 100;
% Settings for Viterbi learning: tempo_tying
%   0) p_n tied across position states (different p_n for each n)
%   1) Global p_n for all changes (only one p_n)
%   2) Separate p_n for tempo increase and decrease (two different p_n)
Params.tempo_tying = 1; 
% Type of transition model and state organisation ('whiteley' or '2015')
Params.transition_model_type = 'whiteley';

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
Params.observationModelType = 'MOG';
% Features (extension) to be used
Params.feat_type{1} = 'lo230_superflux.mvavg';
Params.feat_type{2} = 'hi250_superflux.mvavg';

% DATA:
% =====

% Train data
% ----------

% Train dataset
Params.train_set = 'ballroom';
% Path to lab file
Params.trainLab =  ['~/diss/data/beats/lab_files/', Params.train_set, '.lab'];
% Path to file where bar to rhythm assignments are stored
Params.clusterIdFln = fullfile(Params.data_path, ['ca-', Params.train_set, '-', num2str(length(Params.feat_type)), 'd-', ...
    num2str(Params.R), 'R-meter.mat']);

% Test data
% ----------

% Test dataset
Params.test_set = 'ballroom';
% Path to lab file (.lab) or to test song (.wav)
Params.testLab = ['~/diss/data/beats/lab_files/', Params.test_set, '.lab'];

end
