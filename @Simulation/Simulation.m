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
        function obj = Simulation(config_fun, sim_id, config_path)

            if exist('config_path', 'var')
                old_path = cd(config_path);
                obj.Params = eval(config_fun);
                fprintf('* Reading %s\n', fullfile(config_path, [config_fun, '.m']));
                cd(old_path);
            else
                obj.Params = eval(config_fun);
                fprintf('* Reading %s\n', ['./', config_fun, '.m']);
            end
            
            sys_constructor = str2func(obj.Params.system);
            % create beat tracker object
            obj.system = sys_constructor(obj.Params, sim_id, []);
            % create train_data object
            obj.system.init_train_data(obj.Params);
            % create test_data object
            obj.system.init_test_data(obj.Params);
            % initialize probabilistic model
            obj.system.init_model(obj.Params);
            
            if  obj.Params.n_folds_for_cross_validation > 1
                % do k-fold cross validation: check if lab files for folds are present
                [fpath, fname, ~] = fileparts(obj.Params.testLab);
                for k=1:obj.Params.n_folds_for_cross_validation
                    obj.Params.foldLab{k} = fullfile(fpath, [fname, '-fold', num2str(k), '.lab']);
                    if ~exist(obj.Params.foldLab{k}, 'file')
                        error('Lab file for %i th fold not found: %s', k, obj.Params.foldLab{k});
                    end
                end
                obj.nFolds = obj.Params.n_folds_for_cross_validation;
            elseif obj.Params.n_folds_for_cross_validation == 1
                obj.nFolds = length(obj.system.test_data.file_list);
            elseif obj.Params.n_folds_for_cross_validation == 0
                obj.nFolds = 1;
            else
                error('Parameter n_folds_for_cross_validation is invalid (=%.2f)', obj.Params.n_folds_for_cross_validation)
            end
            obj.save_results2file = 1;
            obj.sim_id = sim_id;
            obj.sim_dir = fullfile(obj.Params.results_path, num2str(sim_id));
            obj.Params.logFileName = num2str(obj.sim_id);
            obj.Params.paramsName = fullfile(obj.sim_dir, 'params.mat');
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
            obj.save_results2file = 1;
        end
        
        function obj = train_system(obj)
                % train model
                obj.system.train_model(obj.Params.init_n_gauss);
        end
               
        function do_sim(obj)
            fileCount = 1;
%             profile on
            for k=1:obj.nFolds
                % train on all except k-th fold
                test_file_ids = obj.retrain(k);
                % do testing
                for iFile=test_file_ids
%                 for iFile=6
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
%             profile viewer
        end
        
        function test_file_ids = retrain(obj, k)
            % Retrain is used to only update parts of the parameters
            
            if obj.Params.n_folds_for_cross_validation == 1 % leave one out
                test_file_ids = k;
                obj.system.retrain_model(test_file_ids);
                
            elseif obj.Params.n_folds_for_cross_validation > 1 % k-fild cross validation
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
            else % no train/test split
                test_file_ids = 1:length(obj.system.test_data.file_list);
            end
            
        end
        
        function results = test(obj, iFile)
            results = obj.system.do_inference(iFile);
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

