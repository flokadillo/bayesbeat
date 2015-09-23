function Results = ex5_train_and_test_pf(in_file, train_files, out_folder)
% [Results] = ex3_train_and_test_hmm(in_file, out_folder)
%   Train an HMM and test it.
% ----------------------------------------------------------------------
% INPUT Parameter:
%   in_file             : audio file
%   out_folder          : folder, where output (beats and model) are stored
%   train_files         : cell array with training audio files
%
% OUTPUT Parameter:
%   Results             : structure array with beat tracking results
%
% 30.07.2015 by Florian Krebs
% ----------------------------------------------------------------------
% Train dataset
% Path to lab file
Params.trainLab = train_files;
Params.testLab = in_file;
Params.data_path = out_folder;
Params.results_path = out_folder;
Params.inferenceMethod = 'PF';
Params.min_tempo_bpm = 100;
Params.max_tempo_bpm = 150;
Params.learn_tempo_ranges = 0;
Params.resampling_scheme = 0;
Params.warp_fun = '@(x) x.^(1/5)';
Params.n_particles = 5000;
Params.ratio_Neff = 0.001; % smaller -> less resampling

% TRAINING THE MODEL

% create beat tracker object
BT = BeatTracker(Params);
% set up test_data
BT.init_test_data();
% train model
BT.train_model();

% TEST THE MODEL
% do beat tracking
Results = BT.do_inference(1);
% save results
[~, fname, ~] = fileparts(Params.testLab);
BT.save_results(Results, out_folder, fname);
end
