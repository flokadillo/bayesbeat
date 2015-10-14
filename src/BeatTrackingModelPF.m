classdef BeatTrackingModelPF < handle & BeatTrackingModel
    % BeatTrackingModelPF class
    % Sub class of BeatTrackingModel which implements Particle Filter
    % related functionality
    
    properties
        PF
        n_particles
        initial_particles
        resampling_params
    end
    
    methods
        function obj = BeatTrackingModelPF(Params, Clustering)
            % Call superclass constructor
            obj@BeatTrackingModel(Params, Clustering);
            State_space_params = obj.parse_params(Params, Clustering);
            State_space_params.patt_trans_opt = Params.patt_trans_opt;
            if State_space_params.use_silence_state
                Clustering.rhythm_names{State_space_params.n_patterns + 1} = ...
                    'silence';
            end
            % Create state_space
            obj.state_space = BeatTrackingStateSpace(...
                State_space_params, Params.min_tempo_bpm, ...
                Params.max_tempo_bpm, Clustering.rhythm2nbeats, ...
                Clustering.rhythm2meter, Params.frame_length, ...
                Clustering.rhythm_names, State_space_params.use_silence_state);
        end
        
        function train_model(obj, transition_probability_params, train_data, ...
                cells_per_whole_note, dist_type, results_path)
            obj.make_initial_distribution(train_data.meters);
            obj.make_transition_model(transition_probability_params);
            obj.make_observation_model(train_data, cells_per_whole_note, ...
                dist_type);
            if ismember(obj.resampling_params.resampling_scheme, [2, 3])
                obj.PF = MixtureParticleFilter(obj.trans_model, obj.obs_model, ...
                    obj.initial_particles, obj.n_particles, ...
                    obj.resampling_params);
            else
                obj.PF = ParticleFilter(obj.trans_model, obj.obs_model, ...
                    obj.initial_particles, obj.n_particles, ...
                    obj.resampling_params);
            end
            fln = fullfile(results_path, 'model.mat');
            pf = obj;
            save(fln, 'pf');
            fprintf('* Saved model (Matlab) to %s\n', fln);
        end
        
        function obj = make_initial_distribution(obj, tempo_per_cluster)
            % TODO: Implement prior initial distribution for tempo
            initial_m = zeros(obj.n_particles, 1);
            initial_n = zeros(obj.n_particles, 1);
            initial_r = zeros(obj.n_particles, 1);
            Meff = obj.state_space.max_position_from_pattern;
            min_tempo_ss = obj.state_space.convert_tempo_from_bpm(...
                obj.state_space.min_tempo_bpm);
            max_tempo_ss = obj.state_space.convert_tempo_from_bpm(...
                obj.state_space.max_tempo_bpm);
            % use pseudo random monte carlo: divide the state space into
            % cells and draw the same number of samples per cell
            n_tempo_cells = floor(sqrt(obj.n_particles / ...
                obj.state_space.n_patterns));
            % number of particles per tempo cell
            n_parts_per_tempo_cell = floor(obj.n_particles / n_tempo_cells);
            % distribute position equally among state space
            pos_cell_size = sum(obj.state_space.max_position_from_pattern) / ...
                n_parts_per_tempo_cell;
            r_m = rand(obj.n_particles, 1) - 0.5; % between -0.5 and +0.5
            r_n = rand(obj.n_particles, 1) - 0.5;
            n_parts_per_pattern = zeros(obj.state_space.n_patterns, 1);
            max_tempo_range = max(max_tempo_ss - min_tempo_ss);
            tempo_cell_size = max_tempo_range / n_tempo_cells;
            c=1;
            for iR = 1:obj.state_space.n_patterns
                % create positions between 1 and Meff(iR)
                m_grid = (0+pos_cell_size/2):pos_cell_size:...
                    (Meff(iR)-pos_cell_size/2);
                n_tempo_cells_iR = round(max([1, n_tempo_cells * ...
                    (max_tempo_ss(iR) - min_tempo_ss(iR)) / max_tempo_range]));
                tempo_grid_iR = linspace(min_tempo_ss(iR), max_tempo_ss(iR), ...
                    n_tempo_cells_iR);
                n_parts_per_pattern(iR) = length(m_grid) * length(tempo_grid_iR);
                % create a tempo x position grid matrix
                position_matrix = repmat(m_grid, length(tempo_grid_iR), 1);
                initial_m(c:c+n_parts_per_pattern(iR)-1) = ...
                    position_matrix(:) + ...
                    r_m(c:c+n_parts_per_pattern(iR)-1) * pos_cell_size + 1;
                tempo_vec = repmat(tempo_grid_iR, 1, length(m_grid))';
                initial_n(c:c+n_parts_per_pattern(iR)-1) = tempo_vec + ...
                    r_n(c:c+n_parts_per_pattern(iR)-1) * tempo_cell_size;
                initial_r(c:c+n_parts_per_pattern(iR)-1) = iR;
                c = c + n_parts_per_pattern(iR);
            end
            if sum(n_parts_per_pattern) < obj.n_particles
                % add remaining particles randomly
                % random pattern assignment
                initial_r(c:end) = round(rand(obj.n_particles+1-c, 1)) ...
                    * (obj.state_space.n_patterns-1) + 1;
                n_between_0_and_1 = r_n(c:end) + 0.5;
                m_between_0_and_1 = r_m(c:end) + 0.5;
                % map n_between_0_and_1 to allowed tempo range
                tempo_range = max_tempo_ss - min_tempo_ss;
                initial_n(c:end) = min_tempo_ss(initial_r(c:end)) + ...
                    n_between_0_and_1 .* tempo_range(initial_r(c:end));
                % map m_between_0_and_1 to allowed position range
                initial_m(c:end) = m_between_0_and_1 .* ...
                    (Meff(initial_r(c:end))) + 1;
            end
            % combine coordinates and add a group id for each particles,
            % which is one in the standard pf
            obj.initial_particles = [initial_m, initial_n, initial_r, ...
                ones(obj.n_particles, 1)];
        end
        
        function [] = make_transition_model(obj, transition_params)
            obj.trans_model = BeatTrackingTransitionModelPF(obj.state_space, ...
                transition_params);
        end
        
        function results = do_inference(obj, y, inference_method, ~)
            if isempty(strfind(inference_method, 'PF'))
                error('Inference method %s not compatible with PF model\n', inference_method);
            end
            % compute observation likelihoods
            obs_lik = obj.obs_model.compute_obs_lik(y);
            [m_path, n_path, r_path] = obj.PF.path_with_best_last_weight(obs_lik);
            results = obj.convert_state_sequence(m_path, n_path, r_path, []);
        end
        
    end
    
    methods (Access = protected)
        function State_space_params = parse_params(obj, Params, Clustering)
            bar_durations = Clustering.rhythm2meter(:, 1) ./ ...
                Clustering.rhythm2meter(:, 2);
            State_space_params.max_positions = bar_durations;
            State_space_params.n_patterns = Params.R;
            if isfield(Params, 'n_particles')
                obj.n_particles = Params.n_particles;
            else
                obj.n_particles = 1000;
            end
            if isfield(Params, 'resampling_scheme')
                obj.resampling_params.resampling_scheme = ...
                    Params.resampling_scheme;
            else
                obj.resampling_params.resampling_scheme = 3;
            end
            if ismember(obj.resampling_params.resampling_scheme, [1, 3]) % APF or AMPF
                if isfield(Params, 'warp_fun')
                    obj.resampling_params.warp_fun = str2func(Params.warp_fun);
                else
                    obj.resampling_params.warp_fun = @(x) x.^(1/4);
                end
            else
                obj.resampling_params.warp_fun = [];
            end
            if ismember(obj.resampling_params.resampling_scheme, [2, 3]) % MPF or AMPF
                if isfield(Params, 'state_distance_coefficients')
                    obj.resampling_params.state_distance_coefficients = ...
                        Params.state_distance_coefficients;
                else
                    obj.resampling_params.state_distance_coefficients = [30, ...
                        1100, 100];
                end
                if isfield(Params, 'cluster_merging_thr')
                    obj.resampling_params.cluster_merging_thr = ...
                        Params.cluster_merging_thr;
                else
                    obj.resampling_params.cluster_merging_thr = 20;
                end
                if isfield(Params, 'cluster_splitting_thr')
                    obj.resampling_params.cluster_splitting_thr = ...
                        Params.cluster_splitting_thr;
                else
                    obj.resampling_params.cluster_splitting_thr = 30;
                end
                if isfield(Params, 'n_initial_clusters')
                    obj.resampling_params.n_initial_clusters = ...
                        Params.n_initial_clusters;
                else
                    obj.resampling_params.n_initial_clusters = 16 * ...
                        State_space_params.n_patterns;
                end
                if isfield(Params, 'n_max_clusters')
                    obj.resampling_params.n_max_clusters = ...
                        Params.n_max_clusters;
                else
                    obj.resampling_params.n_max_clusters = 3 * ...
                        obj.resampling_params.n_initial_clusters;
                end
                if isfield(Params, 'res_int')
                    obj.resampling_params.resampling_interval = Params.res_int;
                else
                    obj.resampling_params.resampling_interval = 30;
                end
                obj.resampling_params.criterion = 'fixed_interval';
            else
                if isfield(Params, 'ratio_Neff')
                    obj.resampling_params.ratio_Neff = Params.ratio_Neff;
                else
                    obj.resampling_params.ratio_Neff = 0.001;
                end
                obj.resampling_params.criterion = 'ESS'; % effective sample size
            end
            State_space_params.use_silence_state = Params.use_silence_state;
            State_space_params.n_patterns = Params.R;
        end

    end
    
end

