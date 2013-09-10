classdef Simulation
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        nFolds              % number of folds (= number of times parameter training is performed)
        sim_id              % id of simulation (= name of folder in the results dir)
        sim_dir             % folder where results are saved
        save_results2file
        Params              % parameters of simulation (from config_bt)
        system              % system that is evaluated (e.g., BeatTracker object)
    end
    
    methods
        function obj = Simulation(config_fun)
            
            obj.Params = eval(config_fun);
            
            sys_constructor = str2func(obj.Params.system);
            % create beat tracker object
            obj.system = sys_constructor(obj.Params);
            % create train_data object
            obj.system = obj.system.init_train_data(obj.Params);
            % create test_data object
            obj.system = obj.system.init_test_data(obj.Params);
            % initialize probabilistic model
            obj.system = obj.system.init_model(obj.Params);
            
            if  obj.Params.doLeaveOneOut > 1
                % do k-fold cross validation: check if lab files for folds are present
                [fpath, fname, ~] = fileparts(obj.Params.testLab);
                for k=1:obj.Params.doLeaveOneOut
                    obj.Params.foldLab{k} = fullfile(fpath, [fname, '-fold', num2str(k), '.lab']);
                    if ~exist(obj.Params.foldLab{k}, 'file')
                        error('Lab file for %i th fold not found: %s', k, obj.Params.foldLab{k});
                    end
                end
                obj.nFolds = obj.Params.doLeaveOneOut;
            elseif obj.Params.doLeaveOneOut == 1
                obj.nFolds = length(obj.system.test_data.file_list);
            elseif obj.Params.doLeaveOneOut == 0
                obj.nFolds = 1;
            else
                error('Parameter doLeaveOneOut is invalid (=%.2f)', obj.Params.doLeaveOneOut)
            end
            obj.save_results2file = 0;
        end
        
        function obj = set_up_results_dir(obj, sim_id)
            obj.sim_id = sim_id;
            % set up folder where results are stored
            obj.sim_dir = fullfile(obj.Params.results_path, num2str(obj.sim_id));
            obj.Params.logFileName = num2str(obj.sim_id);
            obj.Params.paramsName = fullfile(obj.sim_dir, 'params.mat');
            %                 Params.obsFileName = fullfile(obj.sim_dir, 'observationModel.mat');
            %                 Params.transitionMatrixFile = fullfile(obj.sim_dir, 'transitionMatrix.mat');
            % copy config file to simulation folder
            if exist(obj.sim_dir, 'file')
                system(['cp ', fullfile(obj.Params.base_path, 'config_bt.m'), ' ', obj.sim_dir]);
            end
            obj.save_results2file = 1;
        end
        
        function obj = train_system(obj)
            % train model
            obj.system = obj.system.train_model(obj.Params.useTempoPrior);
            
        end
        
        function do_sim(obj)
            fileCount = 1;
            for k=1:obj.nFolds
                % train on all except k-th fold
                test_file_ids = obj.retrain(k);
                % do testing
                for iFile=test_file_ids(:)'
                    [~, fname, ~] = fileparts(obj.system.test_data.file_list{iFile});
                    fprintf('%i/%i) [%i] %s\n', fileCount, length(obj.system.test_data.file_list), iFile, fname);
                    results = obj.test(iFile);
                    if obj.save_results2file
                        % save to file
                        obj.system.save_results(results, obj.sim_dir, fname);
                    end
                    fileCount = fileCount + 1;
                end
            end
        end
        
        function test_file_ids = retrain(obj, k)
            if obj.Params.doLeaveOneOut == 1
                test_file_ids = k;
                obj.system.retrain_model(test_file_ids);
            elseif obj.Params.doLeaveOneOut > 1
                % load lab file of fold and determine indices
                test_file_ids = load(obj.Params.foldLab{k});
                fln = fullfile(obj.Params.data_path, [obj.Params.train_set, '-train_ids.txt']);
                if exist(fln, 'file')
                    % exclude files that are not part of ok_songs
                   ok_songs = load(fln); 
                   idx1 = ismember(ok_songs, test_file_ids);
                   idx2 = cumsum(ones(length(ok_songs), 1));
                   test_file_ids = idx2(idx1);
                end
                obj.system.retrain_model(test_file_ids);
            else
                % no train/test split
                test_file_ids = 1:length(obj.system.test_data.file_list);
            end
            
        end
        
        function results = test(obj, iFile)
            results = obj.system.do_inference(iFile, obj.Params.smoothingWin);
        end
        
        function obj = set_comp_time(obj, comp_time)
            obj.Params.compTime = comp_time;
        end
        
        function save_params(obj)
            if obj.save_results2file
                Params = obj.Params;
                save(obj.Params.paramsName, 'Params');
            end
            
        end
    end
    
end

