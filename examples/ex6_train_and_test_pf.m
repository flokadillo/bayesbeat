function Results = ex6_train_and_test_pf(in_file, train_files, out_folder)
% [Results] = ex6_train_and_test_pf(in_file, out_folder)
%   Train a PF and test it.
% ----------------------------------------------------------------------
% INPUT Parameter:
%   in_file             : audio file
%   out_folder          : folder, where output (beats and model) are stored
%   train_files         : cell array with training audio files
%
% OUTPUT Parameter:
%   Results             : structure array with beat tracking results
%
% 14.10.2015 by Ajay Srinivasamurthy
% ----------------------------------------------------------------------
% Train dataset
% Path to lab file
Params.trainLab = train_files;
Params.testLab = in_file;
Params.data_path = out_folder;
Params.results_path = out_folder;
Params.inferenceMethod = 'PF';
Params.min_tempo_bpm = 60;
Params.max_tempo_bpm = 220;
Params.learn_tempo_ranges = 0;
Params.resampling_scheme = 3; % AMPF
Params.patt_trans_opt = 2; % ISMIR'15
Params.warp_fun = '@(x) x.^(1/10)';
Params.n_particles = 6000;

% TRAINING THE MODEL

% create beat tracker object
BT = BeatTracker(Params);
% train model
BT.train_model();

% TEST THE MODEL
% do beat tracking
Results = BT.do_inference(1);
% save results
[~, fname, ~] = fileparts(Params.testLab);
BT.save_results(Results, out_folder, fname);
end
