classdef Feature
    % feature extracts and loads audio features
    
    properties
        feat_type       % cell array with one cell per feature dimension,
        % containing the extension of the feature file
        % e.g., feat_type{1} = lo230-superflux.mvavg.normz
        feat_dim        % feature dimension
        frame_length    % frame rate in frames per second
        input_fln
        feature
        
    end
    
    methods
        function obj = Feature(feat_type, frame_length)
            if nargin == 0
                obj.feat_type = {'lo230_superflux.mvavg.normZ', 'hi250_superflux.mvavg.normZ'};
                obj.frame_length = 0.02;
            else
                obj.feat_type = feat_type;
                obj.frame_length = frame_length;
            end
            obj.feat_dim = length(feat_type);
        end
        
        function observations = load_feature(obj, input_fln, save_it)
            if exist('save_it', 'var')
                param.save_it = save_it;
            else
                param.save_it = 1; % save feature to folder ./beat_activations
            end
            % parse input_data
            obj.input_fln = input_fln;
            [fpath, fname, ~] = fileparts(input_fln);
            % compute feature from wav ore load it
            detfunc = cell(obj.feat_dim, 1);
            fr = cell(obj.feat_dim, 1);
            for iDim = 1:obj.feat_dim
                fln = fullfile(fpath, 'beat_activations', [fname, '.', obj.feat_type{iDim}]);
                if exist(fln,'file') % load features
                    [detfunc{iDim}, fr{iDim}] = obj.read_activations(fln);
                else % compute features
                    fprintf('    Extracting %s from %s\n', obj.feat_type{iDim}, fname);
                    
                    param.frame_length = obj.frame_length;
                    param.feat_type = obj.feat_type{iDim};
                    % post processing
                    % moving average filter
                    if strfind(obj.feat_type{iDim}, 'mvavg')
                        param.doMvavg = 1;
                    else
                        param.doMvavg = 0;
                    end
                    % normalisation
                    if strfind(obj.feat_type{iDim}, 'normZ')
                        param.norm_each_file = 2; % 2 for z-score computation
                    else
                        param.norm_each_file = 0;
                    end
                    % feature type
                    if strfind(obj.feat_type{iDim}, 'lo230_superflux')
                        param.min_f = 0;
                        param.max_f = 230;
                        [detfunc{iDim}, fr{iDim}] = obj.Compute_Bt_LogFiltSpecFlux(input_fln, param);
                    elseif strfind(obj.feat_type{iDim}, 'hi250_superflux')
                        param.min_f = 250;
                        param.max_f = 44100;
                        [detfunc{iDim}, fr{iDim}] = obj.Compute_Bt_LogFiltSpecFlux(input_fln, param);
                    elseif strfind(obj.feat_type{iDim}, 'superflux')
                        param.min_f = 0;
                        param.max_f = 44100;
                        [detfunc{iDim}, fr{iDim}] = obj.Compute_Bt_LogFiltSpecFlux(input_fln, param);
                    elseif strfind(obj.feat_type{iDim}, 'sprflx')
                        % sebastian's superflux
                        [detfunc{iDim}, fr{iDim}] = obj.python_sprflx(input_fln, param.save_it);
                    else
                        error('Feature %s invalid' ,obj.feat_type{iDim});
                    end
                    
                    
                end
                % adjust framerate of features
                if abs(1/fr{iDim} - obj.frame_length) > 0.001
                    detfunc{iDim} = obj.change_frame_rate(detfunc{iDim}, round(1000*fr{iDim})/1000, 1/obj.frame_length );
                    fr{iDim} = 1/obj.frame_length;
                end
                detfunc{iDim} = detfunc{iDim}(:);
            end
            len = zeros(obj.feat_dim, 1);
            for iDim = 1:obj.feat_dim
                len(iDim) = length(detfunc{iDim});
            end
            if sum(diff(len)) ~= 0
                [len_min, ~] = min(len);
                for iDim = 1:obj.feat_dim
                    detfunc{iDim} = detfunc{iDim}(1:len_min);
                end
            end
            try
                observations = cell2mat(detfunc');
            catch exception
                for iDim=1:obj.feat_dim
                    [m, n] = size(detfunc{iDim})
                end
                error('Error detfunc has strange size!\n');
            end
        end
        
        function observations = load_all_features(obj, file_list)
            n_files = length(file_list);
            observations = cell(n_files, 1);
            for i_file=1:n_files
                observations{i_file} = obj.load_feature(file_list{i_file});
            end
        end
    end
    
    
    methods(Static)
        
        function activations_resampled = change_frame_rate(activations, fr_source, fr_target)
            %   change framerate of feature sequence
            
            % convert time index
            dimension = size(activations, 2);
            len_source = length(activations) / fr_source;
            numframes_target = round(len_source * fr_target);
            framelength_target = 1 / fr_target;
            t = (0:length(activations)-1) / fr_source;
            if abs(fr_source - fr_target) > 0.001
                if (len_source - numframes_target*fr_target) > 0.001 % add samples
                    delta_t = 1/fr_source;
                    num_f = ceil((numframes_target*framelength_target-t(end)) / delta_t);
                    act = 0.5*ones(num_f,1);
                    activations = [activations; act];
                    t = [t t(end)+(1:num_f)*delta_t];
                end
                
                t2 = (0:numframes_target-1)*framelength_target;
                a1 = zeros(numframes_target, dimension);
                for i=1:numframes_target-1
                    a1(i, :) = max(activations((((t-t2(i)) > -0.001) & ((t-t2(i+1)) < -0.001)), :));
                end
                a1(end, :) = mean(activations);
                activations_resampled = a1;
                
            else
                % no conversion needed - return source values:
                activations_resampled = activations;
            end
        end
        
        function [ act, fr ] = read_activations( fln )
            % [ act ] = readactivations( fln )
            %   read sebastian boecks activation file
            % ----------------------------------------------------------------------
            %input parameter:
            % fln           : filename (e.g., media-105907(0.0-10.0).beats)
            %
            %output parameter:
            % bt_act        : activations x 1
            %
            % 4.1.2012 by florian krebs
            % ----------------------------------------------------------------------
            if ~exist(fln,'file')
                act = []; fr = [];
                fprintf('WARNING: %s not found\n', fln);
                return
            end
            fid = fopen(fln,'r');
            c = textscan(fid, '%s %f', 1);
            if strcmpi(c{1},'framerate:')
                fr = c{2};
            else
                fprintf('Warning Feature.read_activations: No FRAMERATE field found\n');
                fr = [];
            end
            c = textscan(fid, '%s %d', 1);
            act = textscan(fid, '%f', c{2});
            act = act{1};
            fclose(fid);
            
            
        end
        
        %         [DetFunc, fr] = Compute_LogFiltSpecFlux(fln, save_it, param);
        
        [DetFunc, fr] = Compute_Bt_LogFiltSpecFlux(wavFileName, param);
        
        [DetFunc, fr] = python_sprflx(wavFileName, save_it);
        
        [S, t, f] = STFT(x, winsize, hopsize, fftsize, fs, type, online, plots, norm);
        
        [ out ] = mvavg( signal, winsize, type );
        
        [] = convert_mat_to_feat_file(fpath, output_ext, input_fr);
        
        %         [DetFunc, fr] = compute_LogFiltSpecFlux2(fln, save_it, param);
    end
    
end

