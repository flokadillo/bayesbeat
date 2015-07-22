function Params = HMM_config(base_path)
% [Params] = HMM_config(base_path)
%   specifies parameters for Meter tracking/inference algorithm using HMM
% ----------------------------------------------------------------------
% INPUT Parameter:
%   base_path  : The base path for data, code and results
%
% OUTPUT Parameter:
%   Params            : structure array with meter tracking parameters
%
% 06.09.2012 by Florian Krebs
% 15.07.2015 modified by Ajay Srinivasamurthy
% ----------------------------------------------------------------------
% Use this file to set all the parameters

%% PRELIMINARIES
% Name the system here
Params.system = 'HMM_MeterTracker';
% The name of the dataset
Params.dataset = 'CMCMDa_small';
% Params.dataset = 'BallroomDataset';
% Params.dataset = 'CretanLeapingDances';
% Params.dataset = 'CMCMDa_v2';
% Params.dataset = 'CMHMDa_small';

% Path settings
if exist('base_path', 'var')
    Params.base_path = base_path;
else
    Params.base_path = '/homedtic/amurthy/UPFWork_Server/PhD';
end
Params.data_path = fullfile(Params.base_path, 'Data', Params.dataset);
Params.results_path = fullfile(Params.base_path, 'BayesResults');
Params.temp_path = fullfile(Params.base_path, 'bayesbeat', 'temp');
if ~isdir(Params.temp_path)
    mkdir(Params.temp_path)
end
%% SIMULATION PARAMETERS:
% Inference and model settings {'HMM_viterbi', 'HMM_forward', 'PF',
% 'PF_viterbi'}
Params.inferenceMethod = 'HMM_viterbi';
Params.store_name = 'HMM';
% If n_depends_on_r=true, then use different tempo limits for each rhythm
% state
Params.n_depends_on_r = 0;
% If patternGiven=true, then take the pattern labels as given
Params.patternGiven = 0;
% n_folds_for_cross_validation
%   0) use train and test set as described below
%   1) use leave-one-out splitting (train and test are the same)
%   k) use k-fold cross-validation (train and test are the same)
Params.n_folds_for_cross_validation = 0;
% If reorganize_bars_into_cluster=true, then reorganise features into
% patterns as given by the cluster_assignment_file. Otherwise, Data.extract_feats_per_file_pattern_barPos_dim 
%is loaded from file.
Params.reorganize_bars_into_cluster = 0; % reorganize in Data.extract_feats_per_file_pattern_barPos_dim
% Number of iterations of Viterbi training (currently only for HMMs)
Params.viterbi_learning_iterations = 0;
% Filename of pre-stored model to load
% Params.model_fln = fullfile(Params.temp_path, 'last_model.mat');
%% SYSTEM PARAMETERS:

% State space size
% ----------------
% Maximum position state (used for the meter with the longest duration)
Params.M = 1600;
% Maximum tempo state 
Params.N = 15;
% Number of rhythmic pattern states
Params.R = 4;

% Number of position grid points per whole note. This is important for the
% observation model, as parameters are tied within this grid.
Params.whole_note_div = 64; 
% Number of grid points of one pattern per meter
% Params.barGrid_eff = Params.whole_note_div * (Params.meters(1, :) ./ Params.meters(2, :)); 
% Length of rhythmic patterns {beat', 'bar'}
Params.pattern_size = 'bar'; % 'beat' or 'bar'
% Audio frame length [sec]
Params.frame_length = 0.02;
% Model initial distribution over tempo states by mixture of init_n_gauss
% Gaussians.
Params.init_n_gauss = 0;

% SILENCE STATES
% ---------------
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

%% HMM parameters
% Tempo
% --------------
% Probability of tempo acceleration (and deceleration)
Params.pn = 0.01;  
% Settings for Viterbi learning: tempo_tying
%   0) p_n tied across position states (different p_n for each n)
%   1) Global p_n for all changes (only one p_n)
%   2) Separate p_n for tempo increase and decrease (two different p_n)
Params.tempo_tying = 1; 
% Squeezing factor for the tempo change distribution in the 2015 TM
%  (higher values prefer a constant tempo over a tempo
%               change from one beat to the next one)
Params.alpha = 100;
% Type of transition model and state organisation ('whiteley' or '2015')
Params.transition_model_type = 'whiteley';
% Learn tempo ranges from data
Params.learn_tempo_ranges = 1;
% Set tempo limits (same for all rhythmic patterns). If 
% learn_tempo_ranges == 1, the tempi learned tempi can be restricted to
% a certain range given by min_tempo and max_tempo [BPM]
% Params.min_tempo = [60 60 60 60];
% Params.max_tempo = [230 230 230 230];
Params.min_tempo = [60 60 60 60];
Params.max_tempo = [230 230 230 230];
% Params.min_tempo = 4;       % Corresponds to about 48 bpm on 4/4, a cycle length of 8 seconds at M = 1600: minN = 4 for Carnatic, 6 for Ballroom, 10 for Cretan
% Params.max_tempo = 15;      % Corresponds to about 180 bpm on 4/4, a cycle length of 2.13 seconds at M = 1600: maxN = 15 for Carnatic, 32 for Ballroom, 24 for Cretan
% When learning tempo ranges, outlier beat intervals can be ignored, e.g.,
% outlier_percentile = 50 uses the median tempo only, = 0 uses all periods
Params.outlier_percentile = 5;

% Rhythmic pattern
% --------------
% Probability of rhythmic pattern change
Params.pr = eye(Params.R);
% Probability of starting from a pattern
Params.prprior = 1/Params.R*ones(1,Params.R);     
% Flag to say how pattern transitions are to be done
Params.patt_trans_opt = 1; 
% 0) No pattern transitions allowed 
% 1) Patterns transitions allowed, using pr and prprior

% Observation model
% -----------------
% Distribution type {invGauss, fixed, gamma, histogram, multivariateHistogram,
% bivariateGauss, mixOfGauss, MOG, MOG3, ...}
Params.observationModelType = 'MOG';
% Features (extension) to be used
Params.feat_type{1} = 'lo230_superflux.mvavg.normZ';
Params.feat_type{2} = 'hi250_superflux.mvavg.normZ';
% Params.feat_type{3} = 'rnnBeatAct.normZ';
% Feature dimension
Params.featureDim = length(Params.feat_type);

%% DATA PARAMETERS: Handled in the wrapper script

% Train data
% ----------
% Train dataset
% Params.train_set = 'train1';
% Path to lab file
% Params.trainLab = 'examples\ex3\trainList.lab';

% Test data
% ----------
% Params.testLab = fullfile(Params.base_path, 'data/audio/05_11007_1-01_Nee_Bhakthi.wav');
% Params.testLab = 'examples\ex3\testList.lab';

%% MUSIC PARAMETERS
% Meters that are modelled by the system, e.g., [9, 3; 8 4] 
% **Carnatic
Params.meters = [3, 5, 7, 8; 4, 8, 8, 4];   % Make sure its in increasing order, bug in code otherwise!
Params.meter_names = {'rupaka', 'kChapu', 'mChapu', 'adi'};
Params.min_tempo = [60 60 60 60];   % Beats per min
Params.max_tempo = [230 230 230 230];  % Beats per min

% **Ballroom
% Params.meters = [4,4,4,4,4,4,3,3; 4,4,4,4,4,4,4,4];   
% Params.meterNames = {'ChaChaCha', 'Jive' , 'Quickstep', 'Rumba' , 'Samba' , 'Tango', 'VienneseWaltz', 'Waltz'};
% Params.min_tempo = [60 60 60 60];
% Params.max_tempo = [230 230 230 230];

% **Hindustani
% Params.meters = [7, 10, 12, 16; 8, 8, 8, 8];   
% Params.meterNames = {'rupak', 'jhap', 'ek', 'teen'};
% Params.min_tempo = [60 60 60 60];
% Params.max_tempo = [230 230 230 230];

% **Cretan
% Params.meters = [2; 4];   
% Params.meterNames = {'cretan'};
% Params.min_tempo = [60];
% Params.max_tempo = [230];
%% OUTPUT PARAMETERS
% If save_inference_data=true, then save complete posterior probability to
% file. This is useful for visualisations.
Params.save_inference_data = 0;
% Correct beat position afterwards by shifting it to a loacl max of the
% onset detection function to correct for the rough discretisation of the
% observation model
Params.correct_beats = 0;
% Stores the pattern cluster figures after kmeans if 1
Params.fig_store_flag = 1;
% Save extracted feature to a folder called "beat_activations" relative to
% the audio folder
Params.save_features_to_file = 1;
% Load pre-computed features from file
Params.load_features_from_file = 1;
% Use mex implementation of viterbi decoding
Params.use_mex_viterbi = 0;
% Save beat times and corresponding position within a bar (.beats.txt)
Params.save_beats = 1;
% Save only downbeats (.downbeats.txt)
Params.save_downbeats = 1;
% Save median tempo (.bpm.txt)
Params.save_tempo = 1;
% Save rhythm (.rhythm.txt)
Params.save_rhythm = 1;
% Save time_signature (.meter.txt)
Params.save_meter = 1;
end
