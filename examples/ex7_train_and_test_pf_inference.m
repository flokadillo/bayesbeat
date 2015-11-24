% Example script to test functionality
clear
close all
clc
% 14.10.2015 by Ajay Srinivasamurthy

% Prelims
addpath('../src/');
basepath = '/media/Code/UPFWork/PhD';
dataset = 'CMCMDa_small';
% train_files = textscan(fopen(trainLabFile,'r'),'%s\n');
% test_files = textscan(fopen(trainLabFile,'r'),'%s\n');
Params.trainLab = fullfile(basepath, 'Data', dataset, 'train_1.lab');
Params.testLab = fullfile(basepath, 'Data', dataset, 'test_1.lab');
Params.meters = [3 4; 5 8; 7 8; 8 4];
Params.meter_names = {'rupaka', 'kChapu', 'mChapu', 'adi'};
out_folder = fullfile(basepath, 'BayesResultsCommon', dataset);
Params.data_path = out_folder;
Params.results_path = out_folder;
Params.inferenceMethod = 'PF';
Params.min_tempo_bpm = 30;
Params.max_tempo_bpm = 110;
Params.learn_tempo_ranges = 0;
Params.resampling_scheme = 3; % AMPF
Params.patt_trans_opt = 1; % Prior trans
Params.warp_fun = '@(x) x.^(1/10)';
Params.n_particles = 6000;
Params.cluster_type = 'kmeans';
Params.n_clusters = 4;
Params.dist_cluster = 'equal';
obj.Params.feat_type = {'lo230_superflux.mvavg.normZ', 'hi250_superflux.mvavg.normZ'};

% TRAINING THE MODEL

% create beat tracker object
BT = BeatTracker(Params);
% train model
BT.train_model();

% TEST THE MODEL
% do beat tracking
testFileIndex = 1;
Results = BT.do_inference(testFileIndex);
% save results
[~, fname, ~] = fileparts(BT.test_data.file_list{testFileIndex});
BT.save_results(Results, out_folder, fname);
