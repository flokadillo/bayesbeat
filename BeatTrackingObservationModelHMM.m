classdef BeatTrackingObservationModelHMM < BeatTrackingObservationModel
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        cell_from_state
        gmm_from_state
    end
    
    methods
        function obj = BeatTrackingObservationModelHMM(state_space, feat_type, ...
                dist_type, cells_per_whole_note)
            % call superclass constructor
            obj@BeatTrackingObservationModel(state_space, feat_type, ...
                dist_type, cells_per_whole_note);
            obj.set_cell_from_state();
            obj.set_gmm_from_state();
        end
        
        function [] = set_cell_from_state(obj)
            positions_per_cell = obj.state_space.max_position(1) ./ ...
                obj.cells_from_pattern(1);
            empty_states = (obj.state_space.pattern_from_state <= 0);
            obj.cell_from_state = floor((obj.state_space.position_from_state ...
                - 1) / positions_per_cell) + 1;
            obj.cell_from_state(empty_states) = nan;
            if obj.state_space.use_silence_state
                % use first cell for silence state
                obj.cell_from_state(end, 2) = 1;
            end
        end
        
        function [] = set_gmm_from_state(obj)
            obj.gmm_from_state = zeros(obj.state_space.n_states, 1);
            for i_s = 1:obj.state_space.n_states
                obj.gmm_from_state(i_s) = ...
                    (obj.state_space.pattern_from_state(i_s) - 1) * ...
                    obj.max_cells + obj.cell_from_state(i_s);
            end
        end
        
        function obs_lik = compute_obs_lik(obj, observations)
            % obs_lik = compute_obs_lik(obj, observations)
            %
            % pre-computes the likelihood of the observations
            %
            % -----------------------------------------------------------------
            %INPUT parameters:
            % observations  : observations [nFrames x nDim]
            %
            %OUTPUT parameters:
            % obslik        : observation likelihood [R * barPos, nFrames]
            %
            % 27.08.2015 by Florian Krebs
            % -----------------------------------------------------------------
            % Compute obs_lik in general format calling the superclass'
            % method
            obs_lik = compute_obs_lik@BeatTrackingObservationModel(obj, ...
                observations);
            % Re-format to [n_patterns * max_cells, n_frames] in order to
            % use it with the general HMM class
            obs_lik = permute(obs_lik,[2 1 3]);
            obs_lik = reshape(obs_lik, [obj.state_space.n_patterns * ...
                obj.max_cells, size(obs_lik, 3)]);
            
        end
    end
    
end

