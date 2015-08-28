classdef BeatTrackingObservationModelHMM < BeatTrackingObservationModel
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        cell_from_state
    end
    
    methods
        function obj = BeatTrackingObservationModelHMM(state_space, feat_type, ...
                dist_type, cells_per_whole_note)
            % call superclass constructor
            obj@BeatTrackingObservationModel(state_space, feat_type, ...
                dist_type, cells_per_whole_note);
            obj.set_cell_from_state();
        end
        
        function [] = set_cell_from_state(obj)
            % Computes state2obs_idx, which specifies which states are tied
            % (share the same parameters)
            % position_state_map    obj.trans_model.mapping_state_position, ...
            % rhythm_state_map      obj.trans_model.mapping_state_rhythm
            positions_per_cell = 1 ./ obj.cells_per_whole_note;
            empty_states = ~(obj.state_space.pattern_from_state > 0);
            obj.cell_from_state = uint32(floor((obj.state_space.position_from_state ...
                - 1) / positions_per_cell) + 1);
            obj.cell_from_state(empty_states) = nan;
            if obj.state_space.use_silence_state
                % use first cell for silence state
                obj.cell_from_state(end, 2) = 1;
            end
        end
    end
    
end

