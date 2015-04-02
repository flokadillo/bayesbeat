function Params = config_hmm_drumotron(base_path)
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
Params.python_madmom_path = '/home/florian/diss/src/python/madmom';

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
% Inference and model settings {'HMM_viterbi', 'HMM_forward', 'HMM_viterbi_lag', 'PF',
% 'PF_viterbi'}
Params.inferenceMethod = 'HMM_forward';
% Number of iterations of Viterbi training (currently only for HMMs)
Params.viterbi_learning_iterations = 0;
% Filename of pre-stored model to load
% Params.model_fln = fullfile(Params.temp_path, 'last_model.mat');
% Params.model_fln = '/home/florian/diss/src/matlab/beat_tracking/bayes_beat/results/80/model.mat';
% Save extracted feature to a folder called "beat_activations" relative to
% the audio folder
Params.save_features_to_file = 1;
Params.use_mex_viterbi = 1;


% SYSTEM PARAMETERS:
% ==================

% State space size
% ----------------

% Maximum position state (used for the meter with the longest duration)
Params.M = 768;
% Maximum tempo state 
Params.N = nan;
% Number of rhythmic pattern states
Params.R = 4;
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
Params.use_silence_state = 1;
% Probability of entering the silence state
Params.p2s = 1e-5; % 0.00001
% Probability of leaving the silence state
Params.pfs = 1e-3; % 0.001
% File from which the silence observation model params are learned
Params.silence_lab = '~/diss/data/beats/lab_files/robo_silence.lab';
% In online mode (forward path), the best state is chosen among a set of
% possible successor state. This set contains position states within a window
% of +/- max_shift frames (default=10)
Params.online.max_shift = 1;
% In online mode, we reset the best state sequence to the global best state
% each update_interval (in audio frames)
Params.online.update_interval = 200;
% To avoid overfitting and prevent the obs_lik to become zero, we set a
% floor (default=1e-7)
Params.online.obs_lik_floor = 0;
% Probability of rhythmic pattern change
Params.pr = 0;
Params.correct_beats = 0;
% Squeezing factor for the tempo change distribution in the 2015 TM
%  (higher values prefer a constant tempo over a tempo
%               change from one beat to the next one)
Params.alpha = 100;
% Set tempo limits (same for all rhythmic patterns). If no ranges are given, they are learned from data.
% Params.min_tempo = 70;
% Params.max_tempo = 100;

% HMM parameters
% --------------

% Probability of tempo acceleration (and deceleration)
Params.pn = 0.01;  
% Settings for Viterbi learning: tempo_tying
%   0) p_n tied across position states (different p_n for each n)
%   1) Global p_n for all changes (only one p_n)
%   2) Separate p_n for tempo increase and decrease (two different p_n)
Params.tempo_tying = 1; 
% Type of transition model and state organisation ('whiteley' or '2015')
Params.transition_model_type = '2015';


% Observation model
% -----------------

% Distribution type {invGauss, fixed, gamma, histogram, multivariateHistogram,
% bivariateGauss, mixOfGauss, MOG, MOG3, ...}
Params.observationModelType = 'MOG';
Params.feat_freq_range = [30, 200; 200, 17000]; 
% Features (extension) to be used
%Params.feat_type{1} = 'superflux_lo_30_Hz_hi_17000_Hz_50_fps.odf';
 Params.feat_type{1} = 'superflux_lo_30_Hz_hi_200_Hz_50_fps.odf';
 Params.feat_type{2} = 'superflux_lo_200_Hz_hi_17000_Hz_50_fps.odf';
 Params.feat_type{3} = 'superflux_lo_200_Hz_hi_17000_Hz_50_fps.odf';
% 
% Params.feat_type{3} = 'superflux_lo_30_Hz_hi_17000_Hz.odf';
% Feature dimension
Params.featureDim = length(Params.feat_type);

% DATA:
% =====

% Train data
% ----------

% Train dataset
Params.train_set = 'cp_guitar_patterns_extended';
% Path to lab file
Params.trainLab =  ['~/diss/data/beats/lab_files/', Params.train_set, '.lab'];
% Path to file where pattern transitions are stored
%  Params.cluster_transitions_fln = fullfile(Params.data_path, ['cluster_transitions-', ...
%      Params.train_set, '-', num2str(Params.featureDim), 'd-', num2str(Params.R), '.txt']);
% Path to file where cluster to bar assignments are stored
Params.clusterIdFln = fullfile(Params.data_path, ['ca-', Params.train_set, '-', num2str(Params.featureDim), 'd-', ...
    num2str(Params.R), 'R-rhythm.mat']);

% Test data
% ----------

% Test dataset
Params.test_set = 'cp_guitar_patterns_extended';
% Path to lab file (.lab) or to test song (.wav)
Params.testLab = ['~/diss/data/beats/lab_files/', Params.test_set, '.lab'];
% Params.testLab = '~/diss/data/beats/cp_guitar_patterns_extended/audio/r3a_filip_160_to_80.wav';
% Params.testLab = '~/diss/data/beats/cp_guitar_patterns_extended/audio/r2a_flo_160_to_80.wav';

end
