classdef BeatTrackingStateSpace < handle
    %UNTITLED3 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        n_patterns
        n_states
        max_n_tempo_states
        n_tempo_states
        n_beats_from_pattern
        meter_from_pattern
        pattern_names
        n_position_states       % cell array with n_patterns cells.
        %                           Each cell contains a vector of length
        %                           n_tempo_states. n_positions are counted per
        %                           beat, i.e., if you need the
        %                           positions per bar, multiply with
        %                           n_beats_from_pattern
        max_position
        position_from_state
        tempo_from_state
        pattern_from_state
        min_tempo_bpm
        max_tempo_bpm
        frame_length
        use_silence_state
        store_proximity         % Store ids of neighborhood states
        proximity_matrix        % Ids of the neighboring states
        %                           for each state [n_states, 6]. This is
        %                           useful for online beat tracking, where
        %                           we search for the best state in the
        %                           neighborhood of the MAP state
    end
    
    methods
        function obj = BeatTrackingStateSpace(n_patterns, max_n_tempo_states, ...
                max_position, min_tempo_bpm, max_tempo_bpm, n_beats_from_pattern, ...
                meter_from_pattern, frame_length, pattern_names, ...
                use_silence_state, store_proximity)
            % store properties
            obj.n_patterns = n_patterns;
            obj.max_n_tempo_states = max_n_tempo_states;
            obj.max_position = max_position;
            obj.min_tempo_bpm = min_tempo_bpm;
            obj.max_tempo_bpm = max_tempo_bpm;
            obj.n_beats_from_pattern = n_beats_from_pattern;
            obj.meter_from_pattern = meter_from_pattern;
            obj.pattern_names = pattern_names;
            obj.frame_length = frame_length;
            obj.use_silence_state = use_silence_state;
            obj.store_proximity = store_proximity;
        end
        
        function [position, tempo, pattern] = decode_state(obj, state)
            % decode state into (position, tempo, pattern)
            position = obj.position_from_state(state);
            tempo = obj.tempo_from_state(state);
            pattern = obj.pattern_from_state(state);
        end
        
        function [bpm] = convert_tempo_to_bpm(obj, tempo)
            pos_per_beat = obj.max_position(1) / obj.n_beats_from_pattern(1);
            bpm = 60 * tempo / (pos_per_beat * obj.frame_length);
        end
        
        function [tempo] = convert_tempo_from_bpm(obj, bpm)
            pos_per_beat = obj.max_position(1) / obj.n_beats_from_pattern(1);
            tempo = bpm * pos_per_beat * obj.frame_length / 60;
        end
    end
    
end

