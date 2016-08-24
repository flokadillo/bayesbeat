function [ ] = ISMIR2016Tracking( input_file_name, output_file_name, ...
    mintempo, maxtempo)
% [ ] = rnn_tracking( input_file_name, output_file_name)
%   Computes beat times of input_file_name and saves them to
%   output_file_name.
% ----------------------------------------------------------------------
%INPUT parameter:
% input_file_name   : path and filenam of the input WAV file
% output_file_name  : path and filename of the output TXT file
%
% DEPENDS ON
% bayes_beat class
%
% 30.08.2015 by Florian Krebs, edited by Andre Holzapfel 4 ISMIR 2016
% ----------------------------------------------------------------------
% get path of function
transition_lambda = 100;
[func_path, ~, ~] = fileparts(mfilename('fullpath'));
addpath(fullfile(func_path, '..', '..'));
[~, fname, fext] = fileparts(input_file_name);
% ---------- SET PARAMETERS --------------------------------------------
Params.observationModelType = 'RNN_db';%observation type that assumes that two network activations are
%provided: one for beats, and one for downbeats, as in AH@ISMIR2016. For
%the slightly different observation model of Sebastian, in addition
%non-beat activations would need to be included
Params.cluster_type = 'meter';
Params.testLab = input_file_name;
Params.feat_type = {strrep(fext, '.', '')};
Params.frame_shift = 1;
Params.use_meter_prior = 0;
fprintf('%i tempo candidates!\n ',length(mintempo));
if length(mintempo)>1
    Params.N=length(mintempo)*7;%multiplier should be odd number
else
    Params.N = 30;%tempo_states;
end
Params.n_clusters = 1;
Params.correct_beats = 1;
Params.learn_tempo_ranges = 0;
Params.pattern_size = 'bar';
Params.alpha = transition_lambda;
Params.min_tempo_bpm = mintempo;%40;
Params.max_tempo_bpm = maxtempo;%300;
Params.whole_note_div = 64;
[Params.results_path, ~, ~] = fileparts(output_file_name);
%added by AH to work on CNN
%Params.feat_type = {'lo230_superflux.mvavg.normZ', 'hi250_superflux.mvavg.normZ'};
Params.feat_type = {'downbeat_cnn_activations', 'beat_cnn_activations'};
Params.trainLab = input_file_name;
Params.frame_length = 1/100;
% ---------- COMPUTE BEATS --------------------------------------------
BT_Simulation('', Params);
end


