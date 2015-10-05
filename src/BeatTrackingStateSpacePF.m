classdef BeatTrackingStateSpacePF < handle & BeatTrackingStateSpace
    % BeatTrackingStateSpacePF class
    % Implements Particle Filter state space
    
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
        function obj = BeatTrackingStateSpacePF(State_space_params, ...
                min_tempo_bpm, max_tempo_bpm, n_beats_from_pattern, ...
                meter_from_pattern, frame_length, pattern_names, ...
                use_silence_state)
            % Call superclass constructor
            obj@BeatTrackingStateSpace(State_space_params, min_tempo_bpm, ...
                max_tempo_bpm, n_beats_from_pattern, ...
                meter_from_pattern, frame_length, pattern_names, ...
                use_silence_state);
        end
    end
    
end

