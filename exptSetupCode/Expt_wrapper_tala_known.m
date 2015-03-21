clear
close all
clc
if ~size(whos('serverFlag'),1)
    serverFlag = 0;        % Default is that we are running on local machine
end
% Server
addpath('../../bayesbeat');
basepath = '/homedtic/amurthy/UPFWork_Server/PhD';
% Laptop
% addpath('../../BayesBeat');
% basepath = 'E:\UPFWork\PhD\Code';
numExp = 3;
folds = 2;
nTalas = 4;
numPatts = [1 2 4 8];
talaIDs = {'rupaka', 'kChapu', 'mChapu', 'adi'};
procLongFiles = 1;      % Set it to 1 if you wish to process longer files
timeFormat = 'HH:MM:SS.FFF dd-mmm-yyyy';
Params.startTime = datestr(clock, timeFormat);
frame_length = 0.02; 
pattern_size = 'bar';
for r = 1:length(numPatts)
    for t = 1:nTalas
        for ex = 1:numExp
            for fld = 1:folds
                % specify a simulation id
                sim_id = 1000*ex+fld;
                % load config file, specify base directory of beat tracker
                % For HMM
%                 Params = HMM_config(basepath);
                % For PF
                Params = PF_config(basepath);
                % Some more parameters to be changed
                Params.meters = Params.meters(:,t);
                Params.meterNames = Params.meterNames(t);
                Params.R = numPatts(r);
                Params.M = Params.M * Params.meters(1)/Params.meters(2);
                % If testing out something, do this!
                Params.store_name = [Params.store_name Params.meterNames{1} filesep 'nPatts_' num2str(Params.R)];
                if Params.inferenceMethod(1:2) == 'HM'
                    disp('An exact inference using HMM chosen');
                elseif Params.inferenceMethod(1:2) == 'PF'
                    fprintf('Approximate inference using a Particle Filter: %s\n', Params.store_name);
                    Params.nParticles = Params.nParticles/nTalas*numPatts(r);
                end
                Params.results_path = fullfile(Params.results_path, Params.dataset,...
                    [Params.inferenceMethod '_' Params.store_name], num2str(sim_id));
                if ~isdir(Params.results_path)
                    mkdir(Params.results_path);
                end
                Params.train_set = ['train_' num2str(fld) '_' Params.meterNames{1}];
                % Path to lab files
                Params.trainLab = fullfile(Params.base_path, 'Data', Params.dataset, ...
                            ['train_' num2str(fld) '_' Params.meterNames{1} '.lab']);
                Params.testLab = fullfile(Params.base_path, 'Data', Params.dataset, ...
                            ['test_' num2str(fld) '_' Params.meterNames{1} '.lab']);
                % CLUSTERING THE DATASET
                data_save_path = Params.results_path;
                Clustering = RhythmCluster(Params.trainLab, Params.feat_type, frame_length,...
                    data_save_path, pattern_size);
                % cluster the dataset according to the meter of each file
                % Params.clusterIdFln = Clustering.make_cluster_assignment_file('meter');
                Clustering.make_feats_per_bar(Params.whole_note_div);
                [Params.clusterIdFln Params.cluster_transitions_fln] = ...
                    Clustering.do_clustering(Params);
                % Params.clusterIdFln = Clustering.do_
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
                    dur(k) = samps(1)/Fs;
                    [~, fname, ~] = fileparts(BT.test_data.file_list{k});
                    timePerFile(k) = NaN;
                    if (dur(k) > 600) && ~procLongFiles    % Piece shorter than 10 min
                        fprintf('File too large: %s\n', BT.test_data.file_list{k})
                    else
                        try
                            results = BT.do_inference(k);
                            timePerFile(k) = results{end};
                            BT.save_results(results, Params.results_path, fname);
                        catch
                            fprintf('Did not process file, some error: %s\n', BT.test_data.file_list{k})
                        end
                    end
                    close all
                    fprintf('%s tala with %d patterns: %s: %d/%d expt, %d/%d fold, %d/%d done...\n',...
                        Params.meterNames{1}, Params.R, fname, ex, numExp, fld,...
                        folds, k, length(BT.test_data.file_list));
                end
                Params.pieceDur{ex,fld} = dur;
                Params.runTime{ex,fld} = timePerFile;
                Params.minN = BT.model.minN;
                Params.maxN = BT.model.maxN;
                Params.Meff = BT.model.Meff;
                clear dur timePerFile;
            end
        end
        [path1, fname, ~] = fileparts(Params.results_path);
        Params.endTime = datestr(clock);
        save(fullfile(path1,'Parameters.mat'),'Params');
        clear Params
    end
end
if serverFlag
    exit;
end