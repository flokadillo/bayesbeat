clear
close all
clc
% Global Basepath
base_path = '~/diss/src/matlab/beat_tracking/bayesbeat-bitbucket';
% Train a HMM model from data
addpath(base_path)
% specify a simulation id
sim_id = 101;
% load config file, specify base directory of beat tracker
Params = ex3_config_bt(base_path);

% CLUSTERING THE DATASET

Clustering = RhythmCluster(Params.trainLab);
% cluster the dataset according to the meter of each file
Params.clusterIdFln = Clustering.make_cluster_assignment_file('meter');

% TRAINING THE MODEL

% create beat tracker object
BT = BeatTracker(Params, sim_id);
% set up training data
BT.init_train_data();
% set up test_data
BT.init_test_data();
% initialize probabilistic model
BT.init_model();
% train model
BT.train_model();

% TEST THE MODEL
% do beat tracking
for k = 1:length(BT.test_data.file_list)
    results = BT.do_inference(k);
    % save results
    [~, fname, ~] = fileparts(BT.test_data.file_list{k});
    BT.save_results(results, fullfile(Params.results_path, num2str(sim_id)), fname);
end