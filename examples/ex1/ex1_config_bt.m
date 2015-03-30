function Params = ex1_config_bt(base_path)
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
% Set path to the bayes_beat class
if exist('base_path', 'var')
    Params.base_path = base_path;
else
    Params.base_path = '~/diss/src/matlab/beat_tracking/bayes_beat';
end
if ~exist(Params.base_path, 'dir')
    error('Please specify path to bayes_beat class in the config file\n');
end
addpath(Params.base_path)
Params.results_path = fullfile(Params.base_path, 'examples/ex1');
Params.temp_path = fullfile(Params.base_path, 'temp');

% SIMULATION PARAMETERS:
% ======================
% Inference and model settings {'HMM_viterbi', 'HMM_forward', 'PF', 'PF_viterbi'}
Params.inferenceMethod = 'HMM_viterbi';
% Filename of pre-stored model to load
Params.model_fln = fullfile(Params.base_path, 'examples/ex1/hmm_boeck.mat');
 
% SYSTEM PARAMETERS:
% ==================
Params.frame_length = 0.02;
 
% Observation model
% -----------------
 Params.feat_type{1} = 'lo230_superflux.mvavg';
 Params.feat_type{2} = 'hi250_superflux.mvavg';

% DATA:
% =====

% Test data
% ----------
 Params.testLab = fullfile(Params.base_path, 'examples/audio/train1.flac');

end
