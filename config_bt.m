function Params = config_bt
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
Params.base_path = '~/diss/src/matlab/beat_tracking/bayes_beat';
Params.data_path = fullfile(Params.base_path, 'data');
Params.results_path = fullfile(Params.base_path, 'results');
Params.temp_path = fullfile(Params.base_path, 'temp');

% SIMULATION PARAMETERS:
% ======================

% If useTempoPrior=true, then apply some non-uniform initial distribution over the
% tempo states
Params.useTempoPrior = 0;
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
% Params.model_fln = fullfile(Params.base_path, 'results/831/hmm-ballroom_train_5-5.mat');

% SYSTEM PARAMETERS:
% ==================

% State space size
% ----------------

% Maximum position state (used for the meter with the longest duration)
Params.M = 1216;
% Maximum tempo state 
Params.N = 23;
% Number of rhythmic pattern states
Params.R = 2;
% Meters that are modelled by the system, e.g., [9, 3; 8 4]
Params.meters = [3, 4; 4, 4]; 
% Maximum position state per rhythmic pattern
Params.Meff = round((Params.meters(1, :) ./ Params.meters(2, :)) * (Params.M ./ max(Params.meters(1, :) ./ Params.meters(2, :))));
% Length of rhythmic patterns {beat', 'bar'}
Params.pattern_size = 'bar'; % 'beat' or 'bar'

% HMM parameters
% --------------

% Probability of tempo acceleration and deceleration
Params.pn = 0.01;  
Params.tempo_tying = 1; % 0 = tempo only tied across position states, 1 = global p_n for all changes, 2 = separate p_n for tempo increase and decrease
%robot
%Params.pattern_size = 'beat'; % 'beat' or 'bar'
%Params.pn = 0.001; 
Params.pr = 0;
Params.pt = 0; % meter change
Params.frame_length = 0.02;
Params.whole_note_div = 64; % number of grid points per whole note
Params.barGrid_eff = Params.whole_note_div * bar_durations; % number of grid points per meter
Params.init_n_gauss = 2;

% PF parameters
% -------------
Params.nParticles = 4000;
Params.sigmaN = 0.0001; % standard deviation
Params.ratio_Neff = 0.02;
Params.resampling_scheme = 3; % 3 = kmeans+apf, 2 = kmeans, 1 = apf, 0 = sisr
Params.state_distance_coefficients = [30, 1, 10];
Params.cluster_merging_thr = 20; % if distance < thr: merge 
Params.cluster_splitting_thr = 30; % if spread > thr: split 
Params.rbpf = 0;
Params.do_viterbi_filtering = 0;
Params.warp_fun = '@(x)x.^(1/4)';
% Params.warp_fun = '@(x)log(10000 * x + 1)';
if strfind(Params.inferenceMethod, 'PF') > 0 
    Params.pn = Params.sigmaN; 
    if Params.resampling_scheme > 1, Params.comment = sprintf('%i-%i-%i-%i-%i', Params.state_distance_coefficients(1), Params.state_distance_coefficients(2), Params.state_distance_coefficients(3), Params.cluster_merging_thr, Params.cluster_splitting_thr); end
    if ismember(Params.resampling_scheme, [0, 2]), Params.warp_fun = ''; end
end


% Observation feature
Params.observationModelType = 'MOG';  % types = {invGauss, fixed, gamma, histogram, multivariateHistogram,
% bivariateGauss, ... mixOfGauss, MOG, MOG3}
Params.feat_type{1} = 'lo230_superflux.mvavg.normZ';
Params.feat_type{2} = 'hi250_superflux.mvavg.normZ';
%      Params.feat_type{1} = 'bt.SF.filtered82.log';
%      Params.feat_type{2} = 'mid250_425_superflux.mvavg.normZ';
%      Params.feat_type{3} = 'hi450_superflux.mvavg.normZ';
% Params.feat_type{1} = 'superflux.mvavg.normZ';
% Params.feat_type{1} = 'sprflx-online';
%      Params.feat_type{1} = 'bt.SF.filtered82.log';
Params.featureDim = length(Params.feat_type);
% make filename where features are stored
Params.featStr = '';
for iDim = 1:Params.featureDim
    featType = strrep(Params.feat_type{iDim}, '.', '-');
    Params.featStr = [Params.featStr, featType];
end



% train data
Params.train_set = 'boeck';
Params.trainLab =  ['~/diss/data/beats/', Params.train_set, '.lab'];
% Params.train_annots_folder = '~/diss/data/beats/ballroom/all';
% Params.clusterIdFln = fullfile(Params.data_path, 'ca_ballroom_8.txt');
Params.clusterIdFln = fullfile(Params.data_path, ['ca-', Params.train_set, '-', num2str(Params.featureDim), 'd-', ...
    num2str(Params.R), '-meter.txt']);
% Params.cluster_transitions_fln = fullfile(Params.data_path, ['cluster_transitions-', ...
%      Params.train_set, '-', num2str(Params.featureDim), 'd-', num2str(Params.R), '.txt']);

% % test data
Params.test_set = 'boeck';
%robot=======
%Params.test_set = 'robo_test';
Params.testLab = ['~/diss/data/beats/', Params.test_set, '.lab'];
% Params.test_set = ' ';
% Params.testLab = '~/diss/data/beats/boeck/train13.wav';
% Params.test_annots_folder =  '~/diss/data/beats/ballroom/all';

[~, clusterFName, ~] = fileparts(Params.clusterIdFln);
% clusterFName = strrep(clusterFName, '-songs', '');
Params.featuresFln = fullfile(Params.data_path, [clusterFName, '_', Params.featStr, '.mat']);
end
