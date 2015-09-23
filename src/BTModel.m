classdef BTModel < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        state_space
        frame_length
        max_bar_cells
        use_silence_state
        correct_beats
        beat_positions
        obs_model
        trans_model
        dist_type
        train_dataset
    end
    
    
    methods
        function obj = BTModel(Params, Clustering)
            obj.parse_params(Params, Clustering);
        end
        
        function beat_positions = get.beat_positions(obj)
            for i_r = 1:obj.state_space.n_patterns
                pos_per_beat = obj.state_space.max_position_from_pattern(i_r) / ...
                    obj.state_space.n_beats_from_pattern(i_r);
                % subtract eps to exclude max_position_from_pattern+1
                obj.beat_positions{i_r} = 1:pos_per_beat:...
                    (obj.state_space.max_position_from_pattern(i_r)+1);
                obj.beat_positions{i_r} = obj.beat_positions{i_r}(1:...
                    obj.state_space.n_beats_from_pattern(i_r));
            end
            beat_positions = obj.beat_positions;
        end
        
        function obj = make_observation_model(obj, train_data, ...
                cells_per_whole_note, dist_type)
            % Create observation model
            obj.obs_model = BeatTrackingObservationModel(obj.state_space, ...
                train_data.feature.feat_type, dist_type, ...
                cells_per_whole_note);
            % Train model
            if ~strcmp(obj.dist_type, 'RNN')
                obj.obs_model = obj.obs_model.train_model(train_data);
                obj.train_dataset = train_data.dataset;
            end
        end
        
        
        
    end
    
    methods (Access=protected)
        function parse_params(obj, Params, Clustering)
            obj.frame_length = Params.frame_length;
            bar_durations = Clustering.rhythm2meter(:, 1) ./ ...
                Clustering.rhythm2meter(:, 2);
            obj.max_bar_cells = max(Params.whole_note_div * bar_durations);
            obj.use_silence_state = Params.use_silence_state;
            if isfield(Params, 'correct_beats')
                obj.correct_beats = Params.correct_beats;
            else
                obj.correct_beats = 0;
            end
        end
        
        function beats = find_beat_times(obj, position_state, rhythm_state, ...
                beat_act)
            % ----------------------------------------------------------------------
            % [beats] = findBeatTimes(position_state, rhythm_state, param_g)
            %   Find beat times from sequence of bar positions of the HMM beat tracker
            % ----------------------------------------------------------------------
            %INPUT parameter:
            % position_state             : sequence of position states
            % rhythm_state               : sequence of rhythm states
            % beat_act                 : beat activation for correction
            %
            %OUTPUT parameter:
            %
            % beats                    : [nBeats x 2] beat times in [sec] and
            %                           bar.beatnumber
            %
            % 29.7.2012 by Florian Krebs
            % ----------------------------------------------------------------------
            n_frames = length(position_state);
            % set up cell array with beat position for each meter
            beatpositions = obj.beat_positions;
            beats = [];
            if obj.correct_beats
                % resolution of observation model in
                % position_states:
                res_obs = max(obj.state_space.max_position_from_pattern)/obj.max_bar_cells;
                [dist, btype] = max(beatpositions{rhythm_state(1)} - ...
                    position_state(1));
                if (abs(dist) < res_obs/2) && (dist < 0)
                    % if beat is shortly before (within res_obs/2) the audio start we
                    % add one beat at the beginning. E.g. m-sequence starts with
                    % m=2
                    % find audioframe that corresponds to beat
                    % position + res_obs
                    j=1;
                    while position_state(j) < ...
                            (beatpositions{rhythm_state(1)}(btype) + res_obs - 1)
                        j = j + 1;
                    end
                    [~, win_max_offset] = max(beat_act(1:j, ...
                        size(beat_act, 2)));
                    beats = [beats; [win_max_offset, btype]];
                end
            end
            for i = 1:n_frames-1
                if rhythm_state(i) == obj.state_space.n_patterns + 1;
                    % silence state
                    continue;
                end
                for j = 1:length(beatpositions{rhythm_state(i)})
                    beat_pos = beatpositions{rhythm_state(i)}(j);
                    beat_detected = false;
                    if position_state(i) == beat_pos;
                        % current frame = beat frame
                        bt = i;
                        beat_detected = true;
                    elseif ((position_state(i+1) > beat_pos) ...
                            && (position_state(i+1) < position_state(i)))
                        % bar transition between frame i and frame i+1
                        bt = interp1([position_state(i); ...
                            obj.state_space.max_position_from_pattern(rhythm_state(i)) + ...
                            position_state(i+1)], [i; i+1], ...
                            obj.state_space.max_position_from_pattern(rhythm_state(i)) + ...
                            beat_pos);
                        beat_detected = true;
                    elseif ((position_state(i) < beat_pos) ...
                            && (position_state(i+1) > beat_pos))
                        % beat position lies between frame i and frame i+1
                        bt = interp1([position_state(i); position_state(i+1)], ...
                            [i; i+1], beat_pos);
                        beat_detected = true;
                    end
                    if beat_detected
                        if obj.correct_beats
                            % find audioframe that corresponds to beat
                            % position + res_obs
                            max_pos=i;
                            while (max_pos < length(position_state)) ...
                                    && (position_state(max_pos) < ...
                                    (beat_pos + res_obs))
                                max_pos = max_pos + 1;
                            end
                            % find max of last observation feature
                            % TODO: specify which feature to use for
                            % correction
                            [~, win_max_offset] = max(beat_act(floor(bt):max_pos, ...
                                size(beat_act, 2)));
                            bt = win_max_offset + i - 1;
                        end
                        % madmom does not use interpolation. This yields
                        % ~round(bt)
                        beats = [beats; [round(bt), j]];
                        break;
                    end
                end
            end
            if ~isempty(beats)
                beats(:, 1) = beats(:, 1) * obj.frame_length;
            end
        end
    end
    
end

