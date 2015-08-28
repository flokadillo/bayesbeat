classdef BeatTrackingStateSpace2015 < handle & BeatTrackingStateSpace
    %UNTITLED5 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
    end
    
    methods
        function obj = BeatTrackingStateSpace2015(n_patterns, max_n_tempo_states, ...
                max_position, min_tempo_bpm, max_tempo_bpm, n_beats_from_pattern, ...
                meter_from_pattern, frame_length, use_silence_state, ...
                store_proximity)
            % Call superclass constructor
            obj@BeatTrackingStateSpace(n_patterns, max_n_tempo_states, ...
                max_position, min_tempo_bpm, max_tempo_bpm, n_beats_from_pattern, ...
                meter_from_pattern, frame_length, use_silence_state, ...
                store_proximity);
            obj.compute_n_position_states();
            obj.n_states = cellfun(@(x) sum(x), obj.n_position_states)' * ...
                obj.n_beats_from_pattern(:);
            if obj.use_silence_state
                obj.n_states = obj.n_states + 1;
            end
            obj.compute_state_mappings();
            obj.n_tempo_states = cellfun(@(x) length(x), obj.n_position_states);
        end
    end
    
    methods (Access=protected)
        function [] = compute_n_position_states(obj)
            % Computes the number of position states that lay between
            % min_tempo and max_tempo for each pattern. The number of
            % position states is per beat, per pattern and per tempo state.
            % ------------------------------------------------------------------
            %OUTPUT parameter:
            %   obj.n_position_states   cell array with n_patterns cells.
            %                           Each cell contains a vector of length
            %                           [n_tempo_states(r), 1] of positions per 
            %                           beat 
            % ------------------------------------------------------------------
            % number of frames per beat (slowest tempo)
            % (python: max_tempo_states)
            max_frames_per_beat = ceil(60 ./ (obj.min_tempo_bpm * ...
                obj.frame_length));
            % number of frames per beat (fastest tempo)
            % (python: mimax_n_tempo_states)
            min_frames_per_beat = floor(60 ./ (obj.max_tempo_bpm * ...
                obj.frame_length));
            % compute number of position states
            obj.n_position_states = cell(obj.n_patterns, 1);
            if isnan(obj.max_n_tempo_states)
                % use max number of tempi and position states:
                for ri=1:obj.n_patterns
                    obj.n_position_states{ri} = ...
                        max_frames_per_beat(ri):-1:min_frames_per_beat(ri);
                end
            else
                % use N tempi and position states and distribute them
                % log2 wise
                for ri=1:obj.n_patterns
                    gridpoints = obj.max_max_n_tempo_states;
                    max_tempi = max_frames_per_beat(ri) - ...
                        min_frames_per_beat(ri) + 1;
                    N_ri = min([max_tempi, obj.max_n_tempo_states]);
                    obj.n_position_states{ri} = ...
                        2.^(linspace(log2(min_frames_per_beat(ri)), ...
                        log2(max_frames_per_beat(ri)), gridpoints));
                    % slowly increase gridpoints, until we have
                    % n_tempo_states tempo states
                    while (length(unique(round(obj.n_position_states{ri}))) < N_ri)
                        gridpoints = gridpoints + 1;
                        obj.n_position_states{ri} = ...
                            2.^(linspace(log2(min_frames_per_beat(ri)), ...
                            log2(max_frames_per_beat(ri)), gridpoints));
                    end
                    % remove duplicates which would have the same tempo
                    obj.n_position_states{ri} = unique(round(...
                        obj.n_position_states{ri}));
                    % reverse order to be consistent with the N=nan
                    % case
                    obj.n_position_states{ri} = ...
                        obj.n_position_states{ri}(end:-1:1);
                end
            end
        end
        
        function [] = compute_state_mappings(obj)
            % compute mapping between state index and (bar position,
            % tempo and rhythmic pattern sub-states)
            obj.position_from_state = ones(obj.n_states, 1) * (-1);
            obj.tempo_from_state = ones(obj.n_states, 1) * (-1);
            obj.pattern_from_state = ones(obj.n_states, 1, 'int32') ...
                * (-1);
            if obj.store_proximity
                obj.proximity_matrix = ones(obj.n_states, 6, ...
                    'int32') * (-1);
            end
            si = 1;
            n_tempo_states = cellfun(@(x) length(x), ...
                obj.n_position_states);
            beat_length = obj.max_position ./ obj.n_beats_from_pattern;
            for ri = 1:obj.n_patterns
                n_pos_states_per_pattern = obj.n_position_states{ri} * ...
                    obj.n_beats_from_pattern(ri);
                for tempo_state_i = 1:n_tempo_states(ri)
                    idx = si:(si+n_pos_states_per_pattern(tempo_state_i)-1);
                    obj.pattern_from_state(idx) = ri;
                    obj.tempo_from_state(idx) = ...
                        beat_length(ri) ./ ...
                        obj.n_position_states{ri}(tempo_state_i);
                    obj.position_from_state(idx) = ...
                        (0:(n_pos_states_per_pattern(tempo_state_i) - 1)) .* ...
                        obj.max_position(ri) ./ ...
                        n_pos_states_per_pattern(tempo_state_i) + 1;
                    if obj.store_proximity
                        % set up proximity matrix
                        % states to the left
                        obj.proximity_matrix(idx, 1) = [idx(end), idx(1:end-1)];
                        % states to the right
                        obj.proximity_matrix(idx, 4) = [idx(2:end), idx(1)];
                        % states to down
                        if tempo_state_i > 1
                            state_id = (0:n_pos_states_per_pattern(tempo_state_i)-1) * ...
                                n_pos_states_per_pattern(tempo_state_i - 1) /  ...
                                n_pos_states_per_pattern(tempo_state_i) + 1;
                            s_start = idx(1) - n_pos_states_per_pattern(tempo_state_i - 1);
                            % left down
                            temp = floor(state_id);
                            % modulo operation
                            temp(temp == 0) = n_pos_states_per_pattern(tempo_state_i - 1);
                            obj.proximity_matrix(idx, 2) = temp + s_start - 1;
                            % right down
                            temp = ceil(state_id);
                            % modulo operation
                            temp(temp > n_pos_states_per_pattern(tempo_state_i - 1)) = 1;
                            obj.proximity_matrix(idx, 3) = temp + s_start - 1;
                            % if left down and right down are equal set to -1
                            obj.proximity_matrix(find(rem(state_id, 1) == 0) + ...
                                + idx(1) - 1, 3) = -1;
                        end
                        % states to up
                        if tempo_state_i < n_tempo_states(ri)
                            % for each position state of the slow tempo find the
                            % corresponding state of the faster tempo
                            state_id = (0:n_pos_states_per_pattern(tempo_state_i)-1) * ...
                                n_pos_states_per_pattern(tempo_state_i + 1) /  ...
                                n_pos_states_per_pattern(tempo_state_i) + 1;
                            % left up
                            temp = floor(state_id);
                            % modulo operation
                            temp(temp == 0) = n_pos_states_per_pattern(tempo_state_i + 1);
                            obj.proximity_matrix(idx, 6) = temp + idx(end);
                            % right up
                            temp = ceil(state_id);
                            % modulo operation
                            temp(temp > n_pos_states_per_pattern(tempo_state_i + 1)) = 1;
                            obj.proximity_matrix(idx, 5) = temp + idx(end);
                            % if left up and right up are equal set to -1
                            obj.proximity_matrix(find(rem(state_id, 1) == 0) + ...
                                idx(1) - 1, 6) = -1;
                        end
                    end
                    si = si + length(idx);
                end
            end
            if obj.use_silence_state
                silence_state_id = obj.n_states;
                obj.tempo_from_state(silence_state_id) = 0;
                obj.position_from_state(silence_state_id) = 0;
                obj.pattern_from_state(silence_state_id) = obj.n_patterns + 1;
            end
        end
        
    end
end

