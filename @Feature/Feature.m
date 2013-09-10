classdef Feature
    % feature extracts and loads audio features
    
    properties
        feat_type       % cell array with one cell per feature dimension,
        % containing the extension of the feature file
        % e.g., feat_type{1} = lo230-superflux.mvavg.normz
        feat_dim        % feature dimension
        frame_length      % frame rate in frames per second
        input_fln
        feature
        
    end
    
    methods
        function obj = Feature(feat_type, frame_length)
            obj.feat_type = feat_type;
            obj.feat_dim = length(feat_type);
            obj.frame_length = frame_length;
        end
        
        
        function obj = load_feature(obj, input_fln)
            % parse input_data
            obj.input_fln = input_fln;
            [fpath, fname, ] = fileparts(input_fln);
            % compute feature from wav
            detfunc = cell(obj.feat_dim, 1);
            fr = cell(obj.feat_dim, 1);
            for iDim = 1:obj.feat_dim
                fln = fullfile(fpath, [fname, '.', obj.feat_type{iDim}]);
                if exist(fln,'file')
                    [detfunc{iDim}, fr{iDim}] = obj.read_activations(fln);
                    % adjust framerate of features
                    if abs(1/fr{iDim} - obj.frame_length) > 0.001
                        detfunc{iDim} = obj.change_frame_rate(detfunc{iDim}, fr{iDim}, 1/obj.frame_length );
                    end
                else
                    fprintf('compute_beats.m: activation file %s not found\n', fname) ;
                    fprintf('(%s)\n', fln) ;
                    return
                end
            end
            obj.feature = cell2mat(detfunc');
        end
    end
    
    methods(Access=protected)
        function obj = compute_bt_feature(obj, input_fln)
            
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
                return
            end
            fid = fopen(fln,'r');
            c = textscan(fid, '%s %f', 1);
            if strcmp(c{1},'framerate:')
                fr = c{2};
            else
                fr = 100;
            end
            c = textscan(fid, '%s %d', 1);
            act = textscan(fid, '%f', c{2});
            act = act{1};
            fclose(fid);
            
            
        end
    end
    
end

