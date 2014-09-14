classdef Data < handle
    % Data Class (represents training and test data)
    properties 
        file_list                       % list of files in the dataset
        lab_fln                         % lab file with list of files of dataset
        bar2file                        % specifies for each bar the file id [nBars x 1]
        bar2cluster                     % specifies for each bar the cluster id [nBars x 1]
        meter                           % meter of each file [nFiles x 2]
        beats                           % beats of each file {nFiles x 1}[n_beats 3]
        n_bars                          % number of bars of each file [nFiles x 1]
        bar_start_id                    % cell [nFiles x 1] [nBeats x 1] with index of first beat of each bar
        full_bar_beats                  % cell [nFiles x 1] [nBeats x 1] 1 = if beat belongs to full bar
        cluster_fln                     % file with cluster id of each bar
        n_clusters                      % total number of clusters
        rhythm_names                    % cell array of rhythmic pattern names
        rhythm2meter                    % specifies for each bar the corresponding meter [R x 2]
        rhythm2meter_state              % specifies for each bar the corresponding meter state [R x 1]
        meter_state2meter               % specifies meter for each meter state [2 x nMeters]
        feats_file_pattern_barPos_dim   % feature values organized by file, pattern, barpos and dim
        feats_silence                   % feature vector of silence
        pattern_size                    % size of one rhythmical pattern {'beat', 'bar'}
        dataset                         % name of the dataset
        barpos_per_frame                % cell array [nFiles x 1] of bar position (1..bar pos 64th grid) of each frame
        pattern_per_frame               % cell array [nFiles x 1] of rhythm of each frame
        feat_type                       % cell array (features (extension) to be used)
    end
    
  
    methods
        
        function obj = Data(lab_fln, train)
            % read lab_fln (a file where all data files are listed)
            % lab_fln:  text file with filenames and path
            % train:    [0, 1] indicates whether 
            if exist(lab_fln, 'file')
                [fpath, obj.dataset, ext] = fileparts(lab_fln);
                if strcmpi(ext, '.lab')
                    fid = fopen(lab_fln, 'r');
                    obj.file_list = textscan(fid, '%s', 'delimiter', '\n'); obj.file_list = obj.file_list{1};
                    fclose(fid);
                    % check excluded files
                    fln = fullfile(fpath, [obj.dataset, '-exclude.txt']);
                    if train && exist(fln, 'file')
                        fid = fopen(fln, 'r');
                        exclude_songs = textscan(fid, '%s');
                        fclose(fid);
                        exclude_songs = exclude_songs{1};
                        fprintf('    Excluding %i songs (listed in %s)\n', length(exclude_songs), fln);
                        obj.file_list = obj.file_list(~ismember(obj.file_list, exclude_songs));
                    end
                else 
                    obj.file_list{1} = lab_fln;
                end
            else
                error('Lab file %s not found\n', lab_fln);
            end
            % replace paths by absolute paths
            for i_file = 1:length(obj.file_list)
                if strcmp(obj.file_list{i_file}(1), '~') || strcmp(obj.file_list{i_file}(1), '/')
                    % ok, absolute path given
                else
                    obj.file_list{i_file} = fullfile(pwd, obj.file_list{i_file});
                end
            end
            obj.lab_fln = lab_fln;
        end
        
        function obj = read_pattern_bars(obj, cluster_fln, pattern_size)
            % read cluster_fln (where cluster ids for each bar in the dataset are specified)
            % and generate obj.bar2file, obj.n_bars, obj.meter_state2meter and obj.rhythm2meter
            % TODO: bar2file, obj.n_bars should be loaded not computed!
            if exist(cluster_fln, 'file')
                C = load(cluster_fln);
                obj.bar2file = C.bar2file;
                obj.n_bars = C.file2nBars;
                obj.rhythm_names = C.rhythm_names;
                obj.bar2cluster = C.bar2rhythm;
                obj.rhythm2meter = C.rhythm2meter;
            else
                error('Cluster file %s not found\n', cluster_fln);
            end
            obj.cluster_fln = cluster_fln;
            if ismember(0, obj.bar2cluster), obj.bar2cluster = obj.bar2cluster + 1; end
            obj.n_clusters = max(obj.bar2cluster);
            
            % read pattern_size
            if exist('pattern_size', 'var')
                obj.pattern_size = pattern_size;
            else
                obj.pattern_size = 'bar';
            end
            
            obj.meter_state2meter = unique(obj.rhythm2meter, 'rows')';
            for iR=1:obj.n_clusters
                for iM=1:size(obj.meter_state2meter, 2)
                    if (obj.meter_state2meter(1, iM) == obj.rhythm2meter(iR, 1)) && (obj.meter_state2meter(2, iM) == obj.rhythm2meter(iR, 2))
                        obj.rhythm2meter_state(iR) = iM;
                        break;
                    end
                end
            end
            if ismember(obj.rhythm2meter_state, 0)
               error('Data.read_pattern_bars: could not assign meter state to rhythm\n'); 
            end
        end
        
        function obj = read_meter(obj)
            obj.meter = zeros(length(obj.file_list), 2);
            meter_files_available = 1;
            for iFile = 1:length(obj.file_list)
                [fpath, fname, ~] = fileparts(obj.file_list{iFile});
                meter_fln = fullfile(strrep(fpath, 'audio', 'annotations/meter'), [fname, '.meter']);
                if exist(meter_fln, 'file')
                    meter = Data.load_annotations_bt(obj.file_list{iFile}, 'meter');
                    if length(meter) == 1
                        obj.meter(iFile, 1) = meter;
                        obj.meter(iFile, 2) = 4;
                        fprintf('Data.read_meter: No denominator in meter file %s found -> adding 4\n', [fname, '.meter']);
                    else
                        obj.meter(iFile, :) = meter;
                    end
                else
                    meter_files_available = 0;
                    fname = meter_fln;
                end
                if ~meter_files_available
                    fprintf('Data.read_meter: One or several meter files not available for this dataset (e.g., %s)\n', fname);
                end
            end
        end
        
        function obj = filter_out_meter(obj, allowed_meters)
            % e.g., allowed_meters = [3, 4]
            if isempty(obj.meter)
                obj = obj.read_meter();
            end
            good_files = ismember(obj.meter, allowed_meters);
            obj.file_list = obj.file_list(good_files);
            obj.meter = obj.meter(good_files);
            if ~isempty(obj.n_bars), obj.n_bars = obj.n_bars(good_files); end
            
            if ~isempty(obj.bar2file)
                bad_files = find(good_files == 0);
                for iFile = 1:length(bad_files)
                    obj.bar2file(obj.bar2file >= bad_files(iFile)) = obj.bar2file(obj.bar2file >= bad_files(iFile)) - 1;
                end
            end
            
            % Check consistency cluster_fln - train_lab
            if sum(obj.n_bars) ~= length(obj.bar2file)
                error('Number of bars not consistent !');
            end
        end
        
        function [tempo_min_per_cluster, tempo_max_per_cluster] = get_tempo_per_cluster(obj)
            tempo_min_per_cluster = NaN(length(obj.file_list), obj.n_clusters);
            tempo_max_per_cluster = NaN(length(obj.file_list), obj.n_clusters);
            for iFile = 1:length(obj.file_list)
%                 [fpath, fname, ~] = fileparts(obj.file_list{iFile});
                [obj.beats{iFile}, error ] = Data.load_annotations_bt(obj.file_list{iFile}, 'beats');
%                 tempo_fln = fullfile(strrep(fpath, 'audio', 'annotations/bpm'), [fname, '.bpm']);
                if error
                    error('Beats file not found\n');
                end
                beat_periods = sort(diff(obj.beats{iFile}(:, 1)), 'descend');
                % ignore the biggest and smallest 10 percent of the beat
                % periods
                beat_periods = beat_periods(max([floor(length(beat_periods)/10), 1]):min([floor(length(beat_periods)*9/10), length(beat_periods)]));
                
                % so far, only the first bar of each file is used and assigned the style to the whole file
                styleId = unique(obj.bar2cluster(obj.bar2file == iFile));
                if ~isempty(styleId)
                    tempo_min_per_cluster(iFile, styleId) = 60/max(beat_periods);
                    tempo_max_per_cluster(iFile, styleId) = 60/min(beat_periods);
%                     tempo_per_cluster(iFile, styleId(1)) = tempo;
                end
            end
	    if sum(isnan(max(tempo_max_per_cluster))) > 0 % cluster without tempo
		error('cluster without bar assignment\n');		
%rhythm_id = find(isnan(tempo_max_per_cluster));
		%for i_r=rhythm_id(:)
	%		file_id = unique(obj.bar2file(obj.bar2cluster==i_r));
%			tempo_min_per_cluster(file_id, i_r) = 60/max(beat_periods);
%                    tempo_max_per_cluster(iFile, styleId(1)) = 60/min(beat_periods);
		%end
	    end
            
        end
        
        function obj = extract_feats_per_file_pattern_barPos_dim(obj, whole_note_div, barGrid_eff, ...
                featureDim, featuresFln, featureType, frame_length, reorganize_bars_into_cluster)
            
            obj.feat_type = featureType;
            % Extract audio features and sort them according to bars and position
            if exist(featuresFln, 'file') && ~reorganize_bars_into_cluster
                load(featuresFln, 'dataPerFile');
            else
                fprintf('* Extract and organise trainings data: \n');
                for iDim = 1:featureDim
                    fprintf('    dim%i\n', iDim);
                    TrainData = Data.extract_bars_from_feature(obj.file_list, ...
                        featureType{iDim}, whole_note_div, frame_length, obj.pattern_size, 1);
                    temp{iDim} = Data.sort_bars_into_clusters(TrainData.dataPerBar, ...
                        obj.bar2cluster, obj.bar2file);
			
                end
                [n_files, ~, bar_grid_max] = size(temp{1});
                dataPerFile = cell(n_files, obj.n_clusters, bar_grid_max, featureDim);
                for iDim = 1:featureDim
                    dataPerFile(:, :, :, iDim) = temp{iDim};
                end

                obj.barpos_per_frame = TrainData.bar_pos_per_frame;
                for i=1:length(TrainData.pattern_per_frame)
                    obj.pattern_per_frame{i} = TrainData.pattern_per_frame{i};
                    obj.pattern_per_frame{i}(~isnan(obj.pattern_per_frame{i})) = obj.bar2cluster(obj.pattern_per_frame{i}(~isnan(obj.pattern_per_frame{i})));
                end
                barpos_per_frame = obj.barpos_per_frame;
                pattern_per_frame = obj.pattern_per_frame;
                save(featuresFln, 'dataPerFile', 'barpos_per_frame', 'pattern_per_frame', '-v7');
                fprintf('    Saved organized features to %s\n', featuresFln);
            end
            obj.feats_file_pattern_barPos_dim = dataPerFile;
            
        end
        
       
        
        function obj = read_beats(obj)
            % reads beats and downbeats from file and stores them in the
            % data object
            for iFile = 1:length(obj.file_list)
                [obj.beats{iFile}, ~ ] = Data.load_annotations_bt(obj.file_list{iFile}, 'beats');
                if strcmp(obj.pattern_size, 'bar')
                    [obj.n_bars(iFile), obj.full_bar_beats{iFile}, obj.bar_start_id{iFile}] = obj.get_full_bars(obj.beats{iFile});
                else
                    obj.n_bars(iFile) = size(obj.beats{iFile}, 1) - 1;
                end
            end
        end
        

        function belief_func = make_belief_functions(obj, model)
            belief_func = cell(length(obj.file_list), 2);
            n_states = model.M * model.N * model.R;
            tol_win_perc_of_ibi = 0.0875;
            %             for i_file = 1:length(obj.file_list)
            for i_file = 1:length(obj.file_list)
		% determine meter state of current file
                t_state = find((obj.meter_state2meter(1, :) == obj.meter(i_file, 1)) &...
                    (obj.meter_state2meter(2, :) == obj.meter(i_file, 2)));
		% determine rhythmic pattern state of current file
                r_state = find(model.rhythm2meter == t_state);
		% determine correct M of current file
                M_i = model.Meff(t_state);
                
		tol_win = floor(ol_win_perc_of_ibi * model.M / obj.meter(i_file, 2));
                % determine beat type of each beat
                btype = round(rem(obj.beats{i_file}(:, 2), 1) * 10); % position of beat in a bar: 1, 2, 3, 4
		% compute bar position m for each beat
                beats_m = (M_i * (btype-1) / max(btype)) + 1;
                % beat frames
                n_beats_i = size(obj.beats{i_file}, 1);
		% pre-allocate memory for rows, cols and values
                i_rows = zeros((tol_win*2+1) * n_beats_i * model.N * length(r_state), 1);
                j_cols = zeros((tol_win*2+1) * n_beats_i * model.N * length(r_state), 1);
                s_vals = ones((tol_win*2+1) * n_beats_i * model.N * length(r_state), 1);
  		% compute inter beat intervals
		ibi_i = diff(obj.beats{i_file}(:, 1));
		ibi_i = [ibi_i; ibi_i(end)];
                for iBeat=1:n_beats_i
		    % -----------------------------------------------------
		    % Variant 1: tolerance win constant in beats over tempi
		    % -----------------------------------------------------
		    m_support = mod((beats_m(iBeat)-tol_win:beats_m(iBeat)+tol_win) - 1, M_i) + 1;
                    m = repmat(m_support, 1, model.N * length(r_state));
                    n = repmat(1:model.N, length(r_state) * length(m_support), 1);
                    r = repmat(r_state(:), model.N * length(m_support), 1);
                    states = sub2ind([model.M, model.N, model.R], m(:), n(:), r(:));
                    idx = (iBeat-1)*(tol_win*2+1)*model.N*length(r_state)+1:(iBeat)*(tol_win*2+1)*model.N*length(r_state);
                    i_rows(idx) = iBeat;
                    j_cols(idx) = states;

		    % -----------------------------------------------------
		    % Variant 2: Tolerance win constant in time over tempi
		    % -----------------------------------------------------
		   % p=1;
		   % for n_i = model.minN(r_state):model.maxN(r_state)
		%	tol_win = n_i * tol_win_perc_of_ibi * ibi_i(iBeat) / model.frame_length;
		%	m_support = mod((beats_m(iBeat)-tol_win:beats_m(iBeat)+tol_win) - 1, M_i) + 1;
		%	states = sub2ind([model.M, model.N, model.R], m_support(:), ones(size(m_support(:)))*n_i, ones(size(m_support(:)))*r_state);
		%	j_cols(p:p+length(states)-1) = states;
		%	i_rows(p:p+length(states)-1) = iBeat;
		%	p = p + length(states);
		 %   end
				
                end
                %                 [~, idx, ~] = unique([i_rows, j_cols], 'rows');
                belief_func{i_file, 1} = round(obj.beats{i_file}(:, 1) / model.frame_length);
                belief_func{i_file, 1}(1) = max([belief_func{i_file, 1}(1), 1]);
                belief_func{i_file, 2} = logical(sparse(i_rows, j_cols, s_vals, n_beats_i, n_states));
            end
        end
        
%         function feats_silence = extract_feature(obj, fln )
%             if exist(fln, 'file')
%                 addpath('~/diss/src/matlab/libs/matlab_utils');
%                 [feats_silence, fr] = readActivations(fln);
%                 if (abs(1/fr - obj.frame_length) > 0.001)
%                     % adjusting framerate:
%                     [ feats_silence ] = ChangeFramerateOfActivations( obj.feats_silence, fr, 1/obj.frame_length );
%                 end
%             else
%                 error('Silence file %s not found\n', feat_fln);
%             end
%         end
        
        
        
    end
    
    methods (Static)
        
        
        
        function make_k_folds(lab_fln, K)
            if exist(lab_fln, 'file')
                fid = fopen(lab_fln, 'r');
                file_list = textscan(fid, '%s'); file_list = file_list{1};
                fclose(fid);
            else
                error('Lab file %s not found\n', lab_fln);
            end
            [fpath, fname, ext] = fileparts(lab_fln);
            %             Indices = crossvalind('Kfold', length(file_list), K);
            C = cvpartition(length(file_list), 'Kfold', K);
            for i=1:K
                fln = fullfile(fpath, [fname, '-fold', num2str(i), ext]);
                dlmwrite(fln, find(test(C, i)), 'delimiter', '\n');
            end
        end
        
        [nBars, beatIdx, barStartIdx] = get_full_bars(beats, tolInt, verbose);
        
        Output = extract_bars_from_feature(source, featExt, barGrid, barGrid_eff, framelength, pattern_size, dooutput);
        
        dataPerFile = sort_bars_into_clusters(dataPerBar, clusterIdx, bar2file);
        
        [ data, error ] = load_annotations_bt( filename, ann_type );
    end
end
