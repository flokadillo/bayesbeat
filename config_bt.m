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
Params.loadFeaturesFromFile = 0;
Params.doTraining = 1;

Params.smoothingWin = 0;
Params.useTempoPrior = 0;
Params.patternGiven = 0;
Params.doLeaveOneOut = 0;
Params.inferenceMethod = 'PF'; % 'HMM_viterbi', 'PF'
% Params.trainObservationModel = 1;
% Params.trainTransitionMatrix = 1;

% System description
% State space size
% Params.M = 2560/1440; % number of discrete position states
% Params.N = 33/19;
Params.M = 1440; % total number of discrete position states (used for the meter with the longest duration)
Params.N = 19;
Params.R = 2;
Params.meters = [9, 8; ...
                 8, 8]; % e.g., [9, 3; 8 4]
Params.T = size(Params.meters, 2);
bar_durations = Params.meters(1, :) ./ Params.meters(2, :);
meter2M = Params.M ./ max(bar_durations);
Params.Meff = round(bar_durations * meter2M);
Params.pn = 0.02;  % 7 for n dependent p_n
Params.pr = 0;
% Params.pr = 1 - 1/Params.R; % probability of change of rhythmic pattern
Params.pt = 0; % meter change
Params.frame_length = 0.02;
% Params.barGrid = 64 * max(bar_durations); % max number of grid points 
% Params.barGrid = 64; % number of grid points per whole note
Params.whole_note_div = 64; % number of grid points per whole note
Params.barGrid_eff = Params.whole_note_div * bar_durations; % number of grid points per meter
Params.init_n_gauss = 2;
Params.nParticles = 1000;
Params.sigmaN = 0.00005;
Params.ratio_Neff = 0.1;
Params.rbpf = 0;

% train data
Params.train_set = 'usul_ah';
Params.trainLab =  ['~/diss/data/beats/', Params.train_set, '.lab'];
% Params.train_annots_folder = '~/diss/data/beats/ballroom/all';
% Params.clusterIdFln = fullfile(Params.data_path, 'ca_ballroom_8.txt');
Params.clusterIdFln = fullfile(Params.data_path, ['ca-', Params.train_set, '-2d-', ...
    num2str(Params.R), '-songs.txt']);
if ~Params.doTraining
    if strfind(Params.inferenceMethod, 'HMM')
        Params.model_fln = fullfile(Params.data_path, ['hmm_', Params.train_set, '.mat']);
    elseif strfind(Params.inferenceMethod, 'PF')
        Params.model_fln = fullfile(Params.data_path, ['pf_', Params.train_set, '.mat']);
    end
end

% % test data
Params.test_set = 'usul_ah';
Params.testLab = ['~/diss/data/beats/', Params.test_set, '.lab'];
% Params.test_annots_folder =  '~/diss/data/beats/ballroom/all';

% Observation feature
Params.observationModelType = 'MOG';  % types = {invGauss, fixed, gamma, histogram, multivariateHistogram,
% bivariateGauss, ... mixOfGauss, MOG, MOG3}
Params.feat_type{1} = 'lo230_superflux.mvavg.normZ';
Params.feat_type{2} = 'hi250_superflux.mvavg.normZ';
%      Params.feat_type{1} = 'lo230_superflux.mvavg.normZ';
%      Params.feat_type{2} = 'mid250_425_superflux.mvavg.normZ';
%      Params.feat_type{3} = 'hi450_superflux.mvavg.normZ';
%      Params.feat_type{1} = 'superflux.mvavg.normZ';
%      Params.feat_type{1} = 'bt.SF.filtered82.log';
Params.featureDim = length(Params.feat_type);
% make filename where features are stored
Params.featStr = '';
for iDim = 1:Params.featureDim
    featType = strrep(Params.feat_type{iDim}, '.', '-');
    Params.featStr = [Params.featStr, featType];
end
[~, clusterFName, ~] = fileparts(Params.clusterIdFln);
clusterFName = strrep(clusterFName, '-songs', '');
Params.featuresFln = fullfile(Params.data_path, [clusterFName, '_', Params.featStr, '.mat']);
end
