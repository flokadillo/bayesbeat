clear
% close all
clc
% Train a HMM model from data
addpath('E:\UPFWork\PhD\Code\BayesBeat')
% specify a simulation id
for ex = 1:2
    sim_id = 1000+ex;
    % load config file, specify base directory of beat tracker
    Params = ex4_config_bt('E:\UPFWork\PhD\Code\BayesBeat');
    Params.train_set = ['train_' num2str(ex)];
    % Path to lab files
    Params.trainLab = ['examples\ex4\train_' num2str(ex) '.lab'];
    Params.testLab = ['examples\ex4\test_' num2str(ex) '.lab'];
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
        [samps Fs] = wavread(BT.test_data.file_list{k},'size');
        results = BT.do_inference(k);
        % save results
        [~, fname, ~] = fileparts(BT.test_data.file_list{k});
        BT.save_results(results, fullfile(Params.results_path, ...
            Params.dataset, Params.resampling_scheme_name{Params.resampling_scheme+1},...
            num2str(sim_id)), fname);
        clear results
        k
        close all
    end
    ex
end