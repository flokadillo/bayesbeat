classdef MixtureParticleFilter < ParticleFilter
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        

    end
    
    methods
        function obj = MixtureParticleFilter(trans_model, obs_model, ...
                initial_particles, n_particles, resampling_params)
            % Call superclass constructor
            obj@ParticleFilter(trans_model, obs_model, initial_particles, ...
                n_particles, resampling_params);
            % divide particles into clusters and store the group id of each
            % particle
            obj.initial_particles(:, 4) = ...
                obj.divide_into_fixed_cells(obj.initial_particles);
        end
        
        function group_id = divide_into_fixed_cells(obj, states)
            % divide space into fixed cells with equal number of grid
            % points for position and tempo states
            % states: [n_particles x nStates]
            % state_dim: [nStates x 1]
            n_cells = obj.resampling_params.n_initial_clusters;
            group_id = zeros(obj.n_particles, 1);
            n_r_bins = obj.state_space.n_patterns;
            n_n_bins = floor(sqrt(n_cells/n_r_bins));
            n_m_bins = floor(n_cells / (n_n_bins * n_r_bins));
            max_pos = max(obj.state_space.max_position_from_pattern);
            min_tempo_ss = obj.state_space.convert_tempo_from_bpm(...
                obj.state_space.min_tempo_bpm);
            max_tempo_ss = obj.state_space.convert_tempo_from_bpm(...
                obj.state_space.max_tempo_bpm);
            min_tempo = min(min_tempo_ss);
            max_tempo = max(max_tempo_ss);
            m_edges = linspace(1, max_pos + 1, n_m_bins + 1);
            n_edges = linspace(min_tempo, max_tempo, n_n_bins + 1);
            % extend lower and upper limit. This prevents that particles
            % fall below the edges when calling histc
            n_edges(1) = 0; n_edges(end) = max_tempo + 1;
            for iR=1:obj.state_space.n_patterns
                % get all particles that belong to pattern iR
                ind = find(states(:, 3) == iR);
                [~, m_cell] = histc(states(ind, 1), m_edges);
                [~, n_cell] = histc(states(ind, 2), n_edges);
                for m = 1:n_m_bins
                    for n=1:n_n_bins
                        ind2 = intersect(ind(m_cell==m), ind(n_cell==n));
                        group_id(ind2) = sub2ind([n_m_bins, n_n_bins, ...
                            obj.state_space.n_patterns], m, n, iR);
                    end
                end
            end
            if sum(group_id==0) > 0
                error('group assignment failed\n')
            end
        end
 
        
    end
    
end

