classdef BeatTrackingStateSpace < handle
    % BeatTrackingStateSpace class
    % Base class that sets up the state space for a beat tracker.
    
    properties
        n_patterns                  % number of patterns
        max_position_from_pattern   % [n_patterns, 1] maximum bar position         
        n_beats_from_pattern        % [n_patterns, 1] number of beats per bar
        meter_from_pattern          % [2, n_patterns] time signature
        frame_length                % audio frame length in [sec]
        min_tempo_bpm               % [n_patterns, 1] min tempo in [bpm]
        %edited in ISMIR2016: min_tempo_bpm can also contain a vector of
        %tempo candidates for reducing the tempo space. Works in connection
        %with HMM2015, and should be included into a separate variable if
        %found useful for a published code version.
        max_tempo_bpm               % [n_patterns, 1] max tempo in [bpm]
        pattern_names               % cell(n_patterns, 1)
        use_silence_state           % true or false
    end 
    
    methods
        function obj = BeatTrackingStateSpace(State_space_params, min_tempo_bpm, ...
                max_tempo_bpm, n_beats_from_pattern, meter_from_pattern, ...
                frame_length, pattern_names, use_silence_state)
            obj.n_patterns = State_space_params.n_patterns;
            obj.max_position_from_pattern = State_space_params.max_positions;
            obj.min_tempo_bpm = min_tempo_bpm;
            obj.max_tempo_bpm = max_tempo_bpm;
            obj.n_beats_from_pattern = n_beats_from_pattern;
            obj.meter_from_pattern = meter_from_pattern;
            obj.pattern_names = pattern_names;
            obj.frame_length = frame_length;
            obj.use_silence_state = use_silence_state;
        end
        
        function [bpm] = convert_tempo_to_bpm(obj, tempo, pattern)
            if nargin == 2
               pattern = (1:obj.n_patterns)';
            end
            pos_per_beat = obj.max_position_from_pattern(pattern) ./ ...
                obj.n_beats_from_pattern(pattern);
            bpm = 60 * tempo(:)' ./ (pos_per_beat(:)' * obj.frame_length);
        end
        
        function [tempo] = convert_tempo_from_bpm(obj, bpm, pattern)
            if nargin == 2
               pattern = (1:obj.n_patterns)';
            end
            pos_per_beat = obj.max_position_from_pattern(pattern) ./ ...
                obj.n_beats_from_pattern(pattern);
            tempo = bpm .* pos_per_beat * obj.frame_length / 60;
        end
    end
    
end

