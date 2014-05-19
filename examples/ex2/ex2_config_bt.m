function Params = ex2_config_bt(base_path)
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
Params.base_path = '~/diss/src/matlab/beat_tracking/bayes_beat';
Params.results_path = fullfile(Params.base_path, 'examples/ex2/results');
Params.temp_path = fullfile(Params.base_path, 'temp');

% SIMULATION PARAMETERS:
% ======================
% Inference and model settings {'HMM_viterbi', 'HMM_forward', 'PF', 'PF_viterbi'}
Params.inferenceMethod = 'PF';
% Filename of pre-stored model to load
% Params.model_fln = fullfile(Params.base_path, 'examples/ex1/hmm_boeck.mat');
Params.model_fln = fullfile(Params.base_path, 'examples/ex2/pf_boeck.mat');
 
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
 Params.testLab = fullfile(Params.base_path, 'examples/audio/train10.flac');

end
