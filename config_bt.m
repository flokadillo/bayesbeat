function Params = config_bt
% [Params] = config_bt
%   specifies Paramseters for beat tracking algorithm
% ----------------------------------------------------------------------
% INPUT Parameter:
%   none
%
% OUTPUT Parameter:
%   Params            : structure array with beat tracking Paramseters
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

% Simulation parameter
% ====================
% useTempoPrior: if 1, then apply some non-uniform initial distribution over the
% tempo states
Params.useTempoPrior = 0;
Params.n_depends_on_r = 0;

% patternGiven: if 1, use the pattern labels as additional input to the
% system
Params.patternGiven = 0;

% n_folds_for_cross_validation: 
Params.n_folds_for_cross_validation = 0;
Params.save_inference_data = 0;
Params.reorganize_bars_into_cluster = 0; % reorganize in Data.extract_feats_per_file_pattern_barPos_dim
Params.inferenceMethod = 'HMM_forward'; % 'HMM_viterbi', 'HMM_forward', 'PF', 'PF_viterbi'
Params.viterbi_learning_iterations = 0;
% Params.trainObservationModel = 1;
% Params.trainTransitionMatrix = 1;

% System description
% State space size

% Params.M = 2560/1440/1216; % number of discrete position states
% Params.N = 47/26/22;
Params.M = 768; % total number of discrete position states (used for the meter with the longest duration)
Params.N = 11;
Params.R = 3;


% Params.M = 480; % total number of discrete position states (used for the meter with the longest duration)
% Params.N = 30;
% Params.R = 1;

Params.meters = [2, 3, 4; 4, 4, 4]; % e.g., [9, 3; 8 4]

% Params.meters = [4; 4];
Params.T = size(Params.meters, 2);


% Params.pattern_size = 'beat'; % 'beat' or 'bar'
% Params.pn = 0.01;  
Params.tempo_tying = 1; % 0 = tempo only tied across position states, 1 = global p_n for all changes, 2 = separate p_n for tempo increase and decrease
%robot
Params.pattern_size = 'bar'; % 'beat' or 'bar'
Params.pn = 0.01; 
Params.pr = 0;
Params.pt = 0; % meter change
Params.use_silence_state = 1;
Params.p2s = 0.00001; % to silence
Params.pfs = 0.001; % from silence
Params.silence_fln{1} = '/home/florian/diss/data/beats/robo_git2/track-silence.wav';
% Params.silence_fln{2} = '/home/florian/diss/data/beats/robo_git2/track-silence-2.wav';
% Params.silence_fln{3} = '/home/florian/diss/data/beats/robo_git2/track-silence-3.wav';
Params.frame_length = 0.02;
Params.whole_note_div = 64; % number of grid points per whole note

Params.init_n_gauss = 2;
Params.max_shift = 6;

% particle filter settings
Params.nParticles = 2000;
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
% Params.feat_type{1} = 'lo230_superflux.mvavg.normZ';
% Params.feat_type{2} = 'hi250_superflux.mvavg.normZ';
%      Params.feat_type{1} = 'bt.SF.filtered82.log';
%      Params.feat_type{2} = 'mid250_425_superflux.mvavg.normZ';
%      Params.feat_type{3} = 'hi450_superflux.mvavg.normZ';
% Params.feat_type{1} = 'superflux.mvavg.normZ';
Params.feat_type{1} = 'sprflx';
%      Params.feat_type{1} = 'bt.SF.filtered82.log';
Params.featureDim = length(Params.feat_type);
% make filename where features are stored
Params.featStr = '';
for iDim = 1:Params.featureDim
    featType = strrep(Params.feat_type{iDim}, '.', '-');
    Params.featStr = [Params.featStr, featType];
end



% train data

Params.train_set = 'robo-all';
% Params.train_set = 'boeck_3_4';

Params.trainLab =  ['~/diss/data/beats/', Params.train_set, '.lab'];
% Params.train_annots_folder = '~/diss/data/beats/ballroom/all';
% Params.clusterIdFln = fullfile(Params.data_path, 'ca_ballroom_8.txt');
Params.clusterIdFln = fullfile(Params.data_path, ['ca-', Params.train_set, '-', num2str(Params.featureDim), 'd-', ...
    num2str(Params.R), '-meter.txt']);
% Params.cluster_transitions_fln = fullfile(Params.data_path, ['cluster_transitions-', ...
%      Params.train_set, '-', num2str(Params.featureDim), 'd-', num2str(Params.R), '.txt']);

% % test data
% Params.test_set = 'ballroom_test_1';
% %robot=======

% Params.test_set = 'robo-all';

% Params.test_set = 'boeck_3_4';
% Params.testLab = ['~/diss/data/beats/', Params.test_set, '.lab'];
Params.test_set = ' ';
Params.testLab = '~/diss/data/beats/robo_git2/test/flo_1.wav';
% Params.test_annots_folder =  viterbi'~/diss/data/beats/ballroom/all';

[~, clusterFName, ~] = fileparts(Params.clusterIdFln);
% clusterFName = strrep(clusterFName, '-songs', '');
Params.featuresFln = fullfile(Params.data_path, [clusterFName, '_', Params.featStr, '.mat']);
end
