classdef BeatTrackingStateSpace2006 < handle & BeatTrackingStateSpace
    %UNTITLED5 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        min_tempo_ss
        max_tempo_ss
        state_from_substate
    end
    
    methods
        function obj = BeatTrackingStateSpace2006(State_space_params, ...
                min_tempo_bpm, max_tempo_bpm, n_beats_from_pattern, ...
                meter_from_pattern, frame_length, pattern_names, ...
                use_silence_state, store_proximity)
            % Call superclass constructor
            obj@BeatTrackingStateSpace(State_space_params, min_tempo_bpm, ...
                max_tempo_bpm, n_beats_from_pattern, ...
                meter_from_pattern, frame_length, pattern_names, ...
                use_silence_state, store_proximity);
            obj.min_tempo_ss = floor(obj.convert_tempo_from_bpm(min_tempo_bpm));
            obj.max_tempo_ss = ceil(obj.convert_tempo_from_bpm(max_tempo_bpm));
            obj.compute_state_mappings();
            obj.max_n_tempo_states = State_space_params.n_tempi;
            obj.n_position_states = State_space_params.max_positions;
        end
    end
    
    methods (Access=protected)
        
        
        function [] = compute_state_mappings(obj)
            num_tempo_states = obj.max_tempo_ss - obj.min_tempo_ss + 1;
            obj.n_states = obj.max_position(:)' * num_tempo_states(:);
            % alloc memory for mappings
            obj.position_from_state = ones(obj.n_states, 1) * (-1);
            obj.tempo_from_state = ones(obj.n_states, 1) * (-1);
            obj.pattern_from_state = ones(obj.n_states, 1) * (-1);
            obj.state_from_substate = ones(max(obj.max_position), ...
                obj.max_n_tempo_states, obj.n_patterns) * (-1);
            counter = 1;
            for rhi = 1:obj.n_patterns
                mi=1:obj.max_position(rhi);
                for ni = obj.min_tempo_ss(rhi):obj.max_tempo_ss(rhi)
                    idx = counter:(counter + obj.max_position(rhi) - 1);
                    % save state mappings
                    obj.position_from_state(idx) = mi;
                    obj.tempo_from_state(idx) = ni;
                    obj.pattern_from_state(idx) = rhi;
                    lin_idx = sub2ind([max(obj.max_position), ...
                        obj.max_n_tempo_states, obj.n_patterns], mi, ...
                        repmat(ni, 1, obj.max_position(rhi)), ...
                        repmat(rhi, 1, obj.max_position(rhi)));
                    obj.state_from_substate(lin_idx) = idx;
                    counter = counter + obj.max_position(rhi);
                    
                end
            end
        end
        
    end
end

