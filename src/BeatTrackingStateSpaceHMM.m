classdef BeatTrackingStateSpaceHMM < handle & BeatTrackingStateSpace
    %UNTITLED3 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        n_states
        max_n_tempo_states
        n_position_states       % cell array with n_patterns cells.
        %                           Each cell contains a vector of length
        %                           n_tempo_states. n_positions are counted per
        %                           beat, i.e., if you need the
        %                           positions per bar, multiply with
        %                           n_beats_from_pattern
        position_from_state
        tempo_from_state
        pattern_from_state
        store_proximity         % Store ids of neighborhood states
        proximity_matrix        % Ids of the neighboring states
        %                           for each state [n_states, 6]. This is
        %                           useful for online beat tracking, where
        %                           we search for the best state in the
        %                           neighborhood of the MAP state
    end
    
    methods
        function obj = BeatTrackingStateSpaceHMM(State_space_params, ...
                min_tempo_bpm, max_tempo_bpm, n_beats_from_pattern, ...
                meter_from_pattern, frame_length, pattern_names, ...
                use_silence_state, store_proximity)
            % Call superclass constructor
            obj@BeatTrackingStateSpace(State_space_params, min_tempo_bpm, ...
                max_tempo_bpm, n_beats_from_pattern, ...
                meter_from_pattern, frame_length, pattern_names, ...
                use_silence_state);
            % store properties
            obj.max_n_tempo_states = State_space_params.n_tempi;
            obj.store_proximity = store_proximity;
            if obj.use_silence_state
                obj.n_states = obj.n_states + 1;
            end
        end
        
        function [position, tempo, pattern] = decode_state(obj, state)
            % decode state into (position, tempo, pattern)
            position = obj.position_from_state(state);
            tempo = obj.tempo_from_state(state);
            pattern = obj.pattern_from_state(state);
        end
        

    end
    
end

