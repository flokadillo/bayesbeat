% clear
close all
clc
if ~size(whos('serverFlag'),1)
    serverFlag = 0;        % Default is that we are running on local machine
    addpath('../../BayesBeat');
    basepath = 'E:\UPFWork\PhD\Code';
else
    % storepwd = pwd;
    % cd('/homedtic/amurthy/UPFWork_Server/PhD/bayesbeat/exptSetupCode');
    addpath('../../bayesbeat');
    basepath = '/homedtic/amurthy/UPFWork_Server/PhD';
end
if ~serverFlag
    % Needs two variables from input script
    talaID = 1;
    ipmp3Path = 'C:\Users\Ajay.ejwt\Copy\CAMUT\Carnatic\Audio\Concerts\Cherthala Ranganatha Sharma\Cherthala Ranganatha Sharma at Arkay\1_01-Varashiki Vahana.mp3';
    opSamaPath = '.\1_01-Varashiki_Vahana.csv';
end
%% Set values for parameters
talas = {'adi', 'rupaka', 'mishraChapu', 'khandaChapu'};
timeSigs = [8 3 7 5; 8 4 8 8];
talaName = talas{talaID};
talaMeter = timeSigs(:,talaID);
frame_length = 0.02; 
pattern_size = 'bar';
Params = PF_config(basepath);
Params.meters = talaMeter;
Params.meterNames = {talaName};
Params.R = 1;
Params.M = Params.M * Params.meters(1)/Params.meters(2);
Params.fig_store_flag = 0;
%% Some more params
segDur = 120;   % Seconds
currDir = pwd;
workDir = fullfile(currDir,'CAMUTdata');
Params.results_path = workDir;
if ~isdir(Params.results_path)
    mkdir(Params.results_path);
end
Params.temp_path = workDir;
Params.nParticles = 6000;
%% Read the mp3 and convert to wav, segment into parts
fprintf('Processing %s...', ipmp3Path);
opwavpath = fullfile(workDir,'temp_full.wav');
str1 = strcat({'lame --decode "'}, ipmp3Path, {'" "'}, opwavpath, '"');
system(str1{1});
[samps Fs] = wavread(opwavpath,'size');
dur = samps(1)/Fs;
numSegments = floor(dur/segDur);
testLabName = fullfile(workDir, 'testFile.lab');
fp = fopen(testLabName,'wt');
for k = 1:numSegments
    startSamp = (k-1)*segDur*Fs + 1;
    if k == numSegments
        lastSamp = samps(1);
    else
        lastSamp = segDur*Fs + startSamp - 1;
    end
    [y Fs] = wavread(opwavpath, [startSamp lastSamp]);
    tempPartName = fullfile(workDir,['temp_' num2str(k) '.wav']);
    wavwrite(y, Fs, tempPartName);
    fprintf(fp,'%s\n',tempPartName);
end
fclose(fp);
Params.train_set = ['train_' Params.meterNames{1}];
%% Path to lab files
Params.trainLab = fullfile(Params.base_path, 'Data', Params.dataset, ...
                            ['CAMUT_' Params.meterNames{1} '.lab']);
Params.testLab = testLabName;
% CLUSTERING THE DATASET
data_save_path = Params.results_path;
Clustering = RhythmCluster(Params.trainLab, Params.feat_type, frame_length,...
       data_save_path, pattern_size);
% cluster the dataset according to the meter of each file
% Params.clusterIdFln = Clustering.make_cluster_assignment_file('meter');
Clustering.make_feats_per_bar(Params.whole_note_div);
[Params.clusterIdFln Params.cluster_transitions_fln] = ...
Clustering.do_clustering(Params);
% TRAINING THE MODEL
% create beat tracker object
BT = BeatTracker(Params, 1001);
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
allSamas = [];
for k = 1:length(BT.test_data.file_list)
    results = BT.do_inference(k);
    beats = results{1};
    sama = beats(beats(:,3) == 1, 1);
    allSamas = [allSamas; sama+(k-1)*segDur];
    fprintf('Done with Part %d/%d...\n', k, length(BT.test_data.file_list));
end
fprintf('Storing output samas to %s...', opSamaPath)
dlmwrite(opSamaPath,allSamas,'precision','%.3f');
fprintf('Cleaning up...')
rmdir(workDir,'s');
if serverFlag
    cd(storepwd);
    exit;
end