function Results = ex3_train_and_test_hmm(in_file, train_files, out_folder)
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
