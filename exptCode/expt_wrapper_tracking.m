% clear
close all
clc
if size(whos('serverFlag'),1) && serverFlag
    % Running on server
    disp('Running on server...')
    addpath('../../bayesbeatflo');
    basepath = '/homedtic/amurthy/UPFWork_Server/PhD';
else
    clear; 
    serverFlag = 0;        % Default is that we are running on local machine
    addpath('../../bayesbeatflo');
    basepath = '/media/Code/UPFWork/PhD';
end
numExp = 3;
folds = 2;
nTalas = 4;
numPatts = [4 2 1];
talaIDs = {'rupaka', 'kChapu', 'mChapu', 'adi'};
% talaIDs = {'rupak', 'jhap', 'ek', 'teen'};
% talaIDs = {'ChaChaCha', 'Jive' , 'Quickstep', 'Rumba' , 'Samba' , 'Tango', 'VienneseWaltz', 'Waltz'};
% talaIDs = {'Cretan'};
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
                % Run config file based on the inference scheme
                % For HMM
                % Params = HMM_config(basepath);
                % For PF
                Params = PF_config(basepath);
                % Some more parameters to be changed
                Params.meters = Params.meters(:,t);
                Params.meterNames = Params.meter_names(t);
                Params.min_tempo = Params.min_tempo(t);
                Params.max_tempo = Params.max_tempo(t);
                Params.R = numPatts(r);
                Params.M = Params.M(t);% * Params.meters(1)/Params.meters(2);
                % Set a name to store the results
                if Params.inferenceMethod(1:2) == 'HM'
                    disp('An exact inference using HMM chosen');
                elseif Params.inferenceMethod(1:2) == 'PF'
                    fprintf('Approximate inference using a Particle Filter: %s\n', Params.store_name);
                    Params.nParticles = 1500*numPatts(r); % Params.nParticles/nTalas*numPatts(r);  % 1500 particles/pattern
                end                
                Params.store_name = [Params.store_name '_6000'];
                Params.results_path = fullfile(Params.results_path, Params.dataset,...
                    'Tracking', Params.store_name, Params.meterNames{1}, ...
                    ['nPatts_' num2str(Params.R)], num2str(sim_id));
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
                Params.clusterIdFln = Clustering.do_clustering(Params.R, ...
                    pattern_size,'meter_names',Params.meter_names, 'save_pattern_fig',...
                    Params.fig_store_flag,'plotting_path',Params.results_path);
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
                dur = zeros(1,length(BT.test_data.file_list));
                timePerFile = NaN(1,length(BT.test_data.file_list));
                for k = 1:length(BT.test_data.file_list)
                    [samps Fs] = wavread(BT.test_data.file_list{k},'size');
                    dur(k) = samps(1)/Fs;
                    [~, fname, ~] = fileparts(BT.test_data.file_list{k});
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
                Params.pieceDur = dur;
                Params.runTime = timePerFile;
                Params.minN = BT.model.minN;
                Params.maxN = BT.model.maxN;
                Params.Meff = BT.model.Meff;
                clear dur timePerFile;
                storeParams{ex,fld} = Params;
            end
        end
        [path1, fname, ~] = fileparts(Params.results_path);
        Params.endTime = datestr(clock);
        save(fullfile(path1,'Parameters.mat'),'storeParams');
        clear Params
    end
end
if serverFlag
    exit;
end
