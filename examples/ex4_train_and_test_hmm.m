function Results = ex4_train_and_test_hmm(in_file, train_files, out_folder)
% Results = ex4_train_and_test_hmm(in_file, train_files, out_folder)
%   Train an HMM and test it. Here, we use the old (2006) transition model
%   proposed in
%   Whiteley, Nick, Ali Taylan Cemgil, and Simon J. Godsill. 
%   Bayesian Modelling of Temporal Structure in Musical Audio.
%   ISMIR. 2006.
% ----------------------------------------------------------------------
% INPUT Parameter:
%   in_file             : audio file
%   out_folder          : folder, where output (beats and model) are stored
%   train_files         : cell array with training audio files
%
% OUTPUT Parameter:
%   Results             : structure array with beat tracking results
%
% 02.09.2015 by Florian Krebs
% ----------------------------------------------------------------------
% Train dataset
% Path to lab file
Params.trainLab = train_files;
Params.testLab = in_file;
Params.data_path = out_folder;
Params.results_path = out_folder;
Params.transition_model_type = '2006';
Params.M = 1200;
Params.N = 30;
% Use kmeans clustering with two clusters
Params.cluster_type = 'kmeans';
Params.n_clusters = 2;

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
