classdef BeatTrackerPF < handle
    % Hidden Markov Model Class
    properties (SetAccess=private)
        PF
        state_space
        trans_model
        pr                          % probability of a switch in rhythmic pattern
        rhythm2meter_state          % assigns each rhythmic pattern to a
        % meter state (1, 2, ...)  - var not needed anymore but keep due to
        % compatibility
        meter_state2meter           % specifies meter for each meter state
        % (9/8, 8/8, 4/4) [2 x nMeters] - var not needed anymore but keep due to
        % compatibility
        barGrid                     % number of different observation model params per bar (e.g., 64)
        max_bar_cells
        minN                        % min tempo (n_min) for each rhythmic pattern
        maxN                        % max tempo (n_max) for each rhythmic pattern
        frame_length                % audio frame length in [sec]
        dist_type                   % type of parametric distribution
        sample_trans_fun            % transition model
        disc_trans_mat              % transition matrices for discrete states
        obs_model                   % observation model
        initial_particles           % initial location of particles 
        %                               [n_particles, 3] in the order
        %                               position, tempo, pattern
        n_particles                  % number of particles
        particles
        particles_grid              % particle grid for viterbi
        sigma_N                     % standard deviation of tempo transition
        resampling_scheme           % type of resampling employed
        do_viterbi_filtering
        save_inference_data         % save intermediate output of particle filter for visualisation
        resampling_params
        inferenceMethod             % 'PF'
        train_dataset                % dataset, which PF was trained on
        pattern_size                % size of one rhythmical pattern {'beat', 'bar'}
        use_silence_state           % state that describes non-musical content
        n_depends_on_r              % Flag to say if tempo depends on the style/pattern/rhythm
    end
    
    methods
        function obj = BeatTrackerPF(Params, Clustering)
            State_space_params = obj.parse_params(Params, Clustering);
            if obj.use_silence_state
                Clustering.rhythm_names{obj.state_space.n_patterns + 1} = ...
                    'silence';
            end
            % Create state_space
            obj.state_space = BeatTrackingStateSpace(...
                State_space_params, Params.min_tempo_bpm, ...
                Params.max_tempo_bpm, Clustering.rhythm2nbeats, ...
                Clustering.rhythm2meter, Params.frame_length, ...
                Clustering.rhythm_names, obj.use_silence_state);
            tempVar = ver('matlab');
            if str2double(tempVar.Release(3:6)) < 2011
                % Matlab 2010
                disp('MATLAB 2010');
                RandStream.setDefaultStream(RandStream('mt19937ar','seed', ...
                    sum(100*clock)));
            else
                rng('default')
                rng('shuffle');
            end
        end
        
        function train_model(obj, transition_probability_params, train_data, ...
                cells_per_whole_note, dist_type, results_path)
            obj.make_initial_distribution(train_data.meters);
            obj.make_transition_model(transition_probability_params);
            obj.make_observation_model(train_data, cells_per_whole_note, ...
                dist_type);
            if ismember(obj.resampling_scheme, [2, 3])
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
        
        
        function obj = make_observation_model(obj, train_data, ...
                cells_per_whole_note, dist_type)
            % Create observation model
            obj.obs_model = BeatTrackingObservationModel(obj.state_space, ...
                train_data.feature.feat_type, dist_type, ...
                cells_per_whole_note);
            % Train model
            if ~strcmp(obj.dist_type, 'RNN')
                obj.obs_model = obj.obs_model.train_model(train_data);
                obj.train_dataset = train_data.dataset;
            end
        end
        
        function [n_new] = sample_tempo(obj, n_old)
            n_new = n_old + randn(obj.n_particles, 1) * obj.sigma_N * obj.M;
            out_of_range = n_new' > obj.maxN(r(:, iFrame));
            n(out_of_range, iFrame) = obj.maxN(r(out_of_range, iFrame));
            out_of_range = n(:, iFrame)' < obj.minN(r(:, iFrame));
            n(out_of_range, iFrame) = obj.minN(r(out_of_range, iFrame));
        end
        
        function obj = retrain_observation_model(obj, data_file_pattern_barpos_dim, pattern_id)
            %{
            Retrains the observation model for all states corresponding to <pattern_id>.

            :param data_file_pattern_barpos_dim: training data
            :param pattern_id: pattern to be retrained. can be a vector, too.
            :returns: the retrained hmm object

            %}
            obj.obs_model = obj.obs_model.retrain_model(data_file_pattern_barpos_dim, pattern_id);
            %             obj.obs_model.learned_params{pattern_id, :} = ...
            %                 obj.obs_model.fit_distribution(data_file_pattern_barpos_dim(:, pattern_id, :, :));
            
        end
        
        function results = do_inference(obj, y, fname, inference_method)
            if isempty(strfind(inference_method, 'PF'))
                error('Inference method %s not compatible with PF model\n', inference_method);
            end
            % compute observation likelihoods
            obs_lik = obj.obs_model.compute_obs_lik(y);
            [m_path, n_path, r_path] = obj.PF.path_with_best_last_weight(obs_lik);
            obj = obj.forward_filtering(obs_lik, fname);
            [m_path, n_path, r_path] = obj.path_via_best_weight();
            
            % strip of silence state
            if obj.use_silence_state
                idx = logical(r_path<=obj.state_space.n_patterns);
            else
                idx = true(length(r_path), 1);
            end
            % compute beat times and bar positions of beats
            meter = zeros(2, length(r_path));
            meter(:, idx) = obj.rhythm2meter(r_path(idx), :)';
            % compute beat times and bar positions of beats
            beats = obj.find_beat_times(m_path, r_path);
            if strcmp(obj.pattern_size, 'bar')
                tempo = meter(1, idx)' .* 60 .* n_path(idx)' ./ ...
                    (obj.Meff(r_path(idx)) * obj.frame_length);
            else
                tempo = 60 .* n_path(idx) / (obj.M * obj.frame_length);
            end
            results{1} = beats;
            results{2} = tempo;
            results{3} = meter;
            results{4} = r_path;
        end
        
        function [ joint_prob ] = compute_joint_of_sequence(obj, state_sequence, obs_lik)
            %obs_lik [R x barPos x nFrames]
            nFrames = size(obs_lik, 3);
            eval_lik = @(x, y) obj.compute_obs_lik(x, y, obs_lik, obj.M / obj.barGrid);
            joint_prob.obslik = zeros(nFrames, 1);
            n_diffs = diff(double(state_sequence(:, 2))); % nDiff(1) = s_s(2) - s_s(1)
            joint_prob.trans = log(normpdf(n_diffs/obj.M, 0, obj.sigma_N));
            for iFrame = 1:nFrames
                joint_prob.obslik(iFrame) = log(eval_lik([state_sequence(iFrame, 1), state_sequence(iFrame, 3)], iFrame));
            end
            joint_prob.sum = sum(joint_prob.trans) + sum(joint_prob.obslik);
        end
        
        function obj = convert_old_model_to_new(obj)
            % check dimensions of member variables. This function might be removed
            % in future, but is important for compatibility with older models
            % (in old models Meff and
            % rhythm2meter_state are row vectors [1 x K] but should be
            % column vectors)
            obj.Meff = obj.Meff(:);
            obj.rhythm2meter_state = obj.rhythm2meter_state(:);
            % In old models, pattern change probability was not saved as
            % matrix [RxR]
            if (length(obj.pr(:)) == 1) && (obj.state_space.n_patterns > 1)
                % expand pr to a matrix [R x R]
                % transitions to other patterns
                pr_mat = ones(obj.state_space.n_patterns, obj.state_space.n_patterns) * (obj.pr / (obj.state_space.n_patterns-1));
                % pattern self transitions
                pr_mat(logical(eye(obj.state_space.n_patterns))) = (1-obj.pr);
                obj.pr = pr_mat;
            end
            if isempty(obj.rhythm2meter)
                obj.rhythm2meter = obj.meter_state2meter(:, ...
                    obj.rhythm2meter_state)';
            end
        end
        
    end
    
    methods (Access=protected)
        
        
        function lik = compute_obs_lik(obj, p_pos, p_pat, iFrame, obslik)
            % states_m_r:   is a [nParts x 2] matrix, where (:, 1) are the
            %               m-values and (:, 2) are the r-values
            % obslik:       likelihood values [R, barGrid, nFrames]
            m_per_grid = obj.state_space.max_position_from_pattern(1) / ...
                obj.obs_model.cells_from_pattern(1);
            p_cell = floor((p_pos - 1) / m_per_grid) + 1;
            obslik = obslik(:, :, iFrame);
            ind = sub2ind([obj.state_space.n_patterns, ...
                obj.obs_model.max_cells], p_pat, p_cell(:));
            lik = obslik(ind);
        end
        
        function [m_path, n_path, r_path] = path_via_best_weight(obj)
            % use particle with highest weight
            % ------------------------------------------------------------
            [~, bestParticle] = max(obj.particles.weight);
            m_path = obj.particles.m(bestParticle, :);
            r_path = obj.particles.r(bestParticle, :);
            n_path = obj.particles.n(bestParticle, :)';
            m_path = m_path(:)';
            n_path = n_path(:)';
            r_path = r_path(:)';
        end
        
        
        function obj = forward_filtering(obj, obs_lik, fname)
            nFrames = size(obs_lik, 3);
            % initialize particles
            iFrame = 1;
            m = zeros(obj.n_particles, nFrames, 'single');
            n = zeros(obj.n_particles, nFrames, 'single');
            r = zeros(obj.n_particles, nFrames, 'single');
            m(:, iFrame) = obj.initial_m;
            n(:, iFrame) = obj.initial_n;
            r(:, iFrame) = obj.initial_r;
            % observation probability
            eval_lik = @(x, y) obj.compute_obs_lik(x, y, z, obs_lik);
            obs = eval_lik(obj.initial_m, obj.initial_r, iFrame);
            weight = log(obs / sum(obs));
            if obj.resampling_scheme > 1
                % divide particles into clusters by kmeans
                groups = obj.divide_into_fixed_cells([m(:, iFrame), ...
                    n(:, iFrame), r(:, iFrame)], [obj.M; obj.N; ...
                    obj.state_space.n_patterns], obj.n_initial_clusters);
                n_clusters = zeros(nFrames, 1);
            else
                groups = ones(obj.n_particles, 1);
            end
            resampling_frames = zeros(nFrames, 1);
            for iFrame=2:nFrames
                % transition from iFrame-1 to iFrame
                m(:, iFrame) = m(:, iFrame-1) + n(:, iFrame-1);
                m(:, iFrame) = mod(m(:, iFrame) - 1, ...
                    obj.Meff(r(:, iFrame-1))) + 1;
                % Pattern transitions to be handled here
                r(:, iFrame) = r(:, iFrame-1);
                % Change the ones for which the bar changed
                crossed_barline = find(m(:, iFrame) < m(:, iFrame-1));
                for rInd = 1:length(crossed_barline)
                    r(crossed_barline(rInd), iFrame) = ...
                        randsample(obj.state_space.n_patterns, 1, true, ...
                        obj.pr(r(crossed_barline(rInd),iFrame-1),:));
                end
                % evaluate particle at iFrame-1
                obs = eval_lik(m(:, iFrame), r(:, iFrame), iFrame);
                weight = weight(:) + log(obs(:));
                % Normalise importance weights
                [weight, ~] = obj.normalizeLogspace(weight');
                % Resampling
                % ------------------------------------------------------------
                if obj.resampling_interval == 0
                    Neff = 1/sum(exp(weight).^2);
                    do_resampling = (Neff < obj.ratio_Neff * obj.n_particles);
                else
                    do_resampling = (rem(iFrame, obj.resampling_interval) == 0);
                end
                if do_resampling && (iFrame < nFrames)
                    resampling_frames(iFrame) = iFrame;
                    if obj.resampling_scheme == 2 || obj.resampling_scheme == 3 % MPF or AMPF
                        groups = obj.divide_into_clusters([m(:, iFrame), ...
                            n(:, iFrame-1), r(:, iFrame)], ...
                            [obj.M; obj.N; obj.state_space.n_patterns], groups);
                        n_clusters(iFrame) = length(unique(groups));
                    end
                    [weight, groups, newIdx] = obj.resample(weight, groups);
                    m(:, 1:iFrame) = m(newIdx, 1:iFrame);
                    r(:, 1:iFrame) = r(newIdx, 1:iFrame);
                    n(:, 1:iFrame) = n(newIdx, 1:iFrame);
                end
                % transition from iFrame-1 to iFrame
                n(:, iFrame) = n(:, iFrame-1) + randn(obj.n_particles, 1) * obj.sigma_N * obj.M;
                out_of_range = n(:, iFrame)' > obj.maxN(r(:, iFrame));
                n(out_of_range, iFrame) = obj.maxN(r(out_of_range, iFrame));
                out_of_range = n(:, iFrame)' < obj.minN(r(:, iFrame));
                n(out_of_range, iFrame) = obj.minN(r(out_of_range, iFrame));
            end
            obj.particles.m = m;
            obj.particles.n = n;
            obj.particles.r = r;
            obj.particles.weight = weight;
            fprintf('    Average resampling interval: %.2f frames\n', ...
                mean(diff(resampling_frames(resampling_frames>0))));
            if obj.resampling_scheme > 1
                fprintf('    Average number of clusters: %.2f frames\n', mean(n_clusters(n_clusters>0)));
            end
            if obj.save_inference_data
                save(['./', fname, '_pf.mat'], 'logP_data_pf');
            end
        end
        
        function beats = find_beat_times(obj, positionPath, rhythmPath)
            %   Find beat times from sequence of bar positions of the HMM beat tracker
            % ----------------------------------------------------------------------
            %INPUT parameter:
            % positionPath             : sequence of position states
            % rhythmPath                : sequence of meter states
            %                           NOTE: so far only median of sequence is used !
            % nBarPos                  : bar length in bar positions (=M)
            % framelength              : duration of being in one state in [sec]
            %
            %OUTPUT parameter:
            %
            % beats                    : [nBeats x 2] beat times in [sec] and
            %                           beatnumber
            %
            % 29.7.2012 by Florian Krebs
            % ----------------------------------------------------------------------
            numframes = length(positionPath);
            beatpositions = cell(obj.state_space.n_patterns, 1);
            for i_r=1:obj.state_space.n_patterns
                is_compund_meter = ismember(obj.rhythm2meter(i_r, 1), ...
                    [6, 9, 12]);
                if is_compund_meter
                    beatpositions{i_r} = round(linspace(1, obj.Meff(i_r), ...
                        obj.rhythm2meter(i_r, 1) / 3 + 1));
                else % simple meter
                    beatpositions{i_r} = round(linspace(1, obj.Meff(i_r), ...
                        obj.rhythm2meter(i_r, 1) + 1));
                end
                beatpositions{i_r} = beatpositions{i_r}(1:end-1);
            end
            beatno = [];
            beats = [];
            for i = 1:numframes-1
                if rhythmPath(i) == 0
                    continue;
                end
                for beat_pos = 1:length(beatpositions{rhythmPath(i)})
                    if positionPath(i) == beatpositions{rhythmPath(i)}(beat_pos)
                        % current frame = beat frame
                        beats = [beats; [i, beat_pos]];
                        break;
                    elseif ((positionPath(i+1) > beatpositions{rhythmPath(i)}(beat_pos)) && (positionPath(i+1) < positionPath(i)))
                        % bar transition between frame i and frame i+1
                        bt = interp1([positionPath(i); obj.M+positionPath(i+1)],[i; i+1],obj.M+beatpositions{rhythmPath(i)}(beat_pos));
                        beats = [beats; [round(bt), beat_pos]];
                        break;
                    elseif ((positionPath(i) < beatpositions{rhythmPath(i)}(beat_pos)) && (beatpositions{rhythmPath(i)}(beat_pos) < positionPath(i+1)))
                        % beat position lies between frame i and frame i+1
                        bt = interp1([positionPath(i); positionPath(i+1)],[i; i+1],beatpositions{rhythmPath(i)}(beat_pos));
                        beats = [beats; [round(bt), beat_pos]];
                        break;
                    end
                end
            end
            beats(:, 1) = beats(:, 1) * obj.frame_length;
        end
        
        function [groups] = divide_into_clusters(obj, states, state_dims, groups_old)
            % states: [n_particles x nStates]
            % state_dim: [nStates x 1]
            % groups_old: [n_particles x 1] group labels of the particles
            %               after last resampling step (used for initialisation)
            warning('off');
            [group_ids, ~, IC] = unique(groups_old);
            %             fprintf('    %i groups >', length(group_ids));
            k = length(group_ids); % number of clusters
            
            % adjust the range of each state variable to make equally
            % important for the clustering
            points = zeros(obj.n_particles, length(state_dims)+1);
            points(:, 1) = (sin(states(:, 1) * 2 * pi ./ ...
                obj.Meff(states(:, 3))) + 1) * ...
                obj.state_distance_coefficients(1);
            points(:, 2) = (cos(states(:, 1) * 2 * pi ./ ...
                obj.Meff(states(:, 3))) + 1) * ...
                obj.state_distance_coefficients(1);
            points(:, 3) = states(:, 2) * ...
                obj.state_distance_coefficients(2);
            points(:, 4) =(states(:, 3)-1) * ...
                obj.state_distance_coefficients(3) + 1;
            
            % compute centroid of clusters
            % TODO: vectorise!
            centroids = zeros(k, length(state_dims)+1);
            for i_dim=1:size(points, 2)
                %                 centroids(iCluster, :) = mean(points(groups_old == group_ids(iCluster) , :));
                centroids(:, i_dim) = accumarray(IC, points(:, i_dim), [], @mean);
            end
            % do k-means clustering
            options = statset('MaxIter', 1);
            [groups, centroids, total_dist_per_cluster] = kmeans(points, k, 'replicates', 1, ...
                'start', centroids, 'emptyaction', 'drop', 'Distance', 'sqEuclidean', 'options', options);
            %             remove empty clusters
            total_dist_per_cluster = total_dist_per_cluster(~isnan(centroids(:, 1)));
            centroids = centroids(~isnan(centroids(:, 1)), :);
            [group_ids, ~, j] = unique(groups);
            group_ids = 1:length(group_ids);
            groups = group_ids(j)';
            % check if centroids are too close
            merging = 1;
            merged = 0;
            while merging
                D = squareform(pdist(centroids, 'euclidean'), 'tomatrix');
                ind = (tril(D, 0) > 0);
                D(ind) = nan;
                D(logical(eye(size(centroids, 1)))) = nan;
                
                % find minimum distance
                [min_D, arg_min] = min(D(:));
                if min_D > obj.cluster_merging_thr,
                    merging = 0;
                else
                    [c1, c2] = ind2sub(size(D), arg_min);
                    groups(groups==c2(1)) = c1(1);
                    % new centroid
                    centroids(c1(1), :) = mean(points(groups==c1(1), :));
                    % squared Euclidean distance
                    total_dist_per_cluster(c1(1)) = sum(sum(bsxfun(@minus, points(groups==c1(1), :), centroids(c1(1), :)).^2));
                    if length(c1) == 1,  merging = 0;  end
                    % remove old centroid
                    centroids = centroids([1:c2(1)-1, c2(1)+1:end], :);
                    total_dist_per_cluster = total_dist_per_cluster([1:c2(1)-1, c2(1)+1:end]);
                    merged = 1;
                end
            end
            % check if cluster spread is too high
            split = 0;
            [group_ids, ~, j] = unique(groups);
            group_ids = 1:length(group_ids);
            groups = group_ids(j)';
            n_parts_per_cluster = hist(groups, 1:length(group_ids));
            separate_cl_idx = find((total_dist_per_cluster ./ n_parts_per_cluster') > obj.cluster_splitting_thr);
            %             separate_cl_idx = find(mean_dist_per_cluster > obj.cluster_splitting_thr);
            for iCluster = 1:length(separate_cl_idx)
                % find particles that belong to the cluster to split
                parts_idx = find((groups == separate_cl_idx(iCluster)));
                % put second half into a new group
                groups(parts_idx(round(length(parts_idx)/2)+1:end)) = max(groups) + 1;
                % update centroid
                centroids(separate_cl_idx(iCluster), :) = mean(points(parts_idx(1:round(length(parts_idx)/2)), :), 1);
                % add new centroid
                centroids = [centroids; mean(points(parts_idx(round(length(parts_idx)/2)+1:end), :), 1)];
                split = 1;
            end
            if split || merged
                try
                    [groups, ~, ~] = kmeans(points, [], 'replicates', 1, 'start', centroids, 'emptyaction', 'drop', ...
                        'Distance', 'sqEuclidean', 'options', options);
                catch exception
                    centroids
                    error('Problem\n');
                end
                [group_ids, ~, j] = unique(groups);
                group_ids = 1:length(group_ids);
                groups = group_ids(j)';
            end
            warning('on');
        end
        
        
        function [weight, groups, newIdx] = resample(obj, weight, groups)
            if obj.resampling_scheme == 0
                newIdx = obj.resampleSystematic(exp(weight));
                weight = log(ones(obj.n_particles, 1) / obj.n_particles);
            elseif obj.resampling_scheme == 1 % APF
                % warping:
                w = exp(weight);
                f = str2func(obj.warp_fun);
                w_warped = f(w);
                newIdx = obj.resampleSystematic(w_warped);
                w_fac = w ./ w_warped;
                weight = log( w_fac(newIdx) / sum(w_fac(newIdx)) );
            elseif obj.resampling_scheme == 2 % K-MEANS
                % k-means clustering
                [newIdx, weight, groups] = obj.resample_in_groups(groups, ...
                    weight, obj.n_max_clusters);
                weight = weight';
            elseif obj.resampling_scheme == 3 % APF + K-MEANS
                % apf and k-means
                [newIdx, weight, groups] = obj.resample_in_groups(groups, ...
                    weight, obj.n_max_clusters, str2func(obj.warp_fun));
                weight = weight';
            else
                fprintf('WARNING: Unknown resampling scheme!\n');
            end
        end
        
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
            obj.frame_length = Params.frame_length;
            
            if isfield(Params, 'resampling_scheme')
                obj.resampling_scheme = Params.resampling_scheme;
            else
                obj.resampling_scheme = 3;
            end
            if ismember(obj.resampling_scheme, [1, 3]) % APF or AMPF
                if isfield(Params, 'warp_fun')
                    obj.resampling_params.warp_fun = Params.warp_fun;
                else
                    obj.resampling_params.warp_fun = '@(x)x.^(1/4)';
                end
            end
            if ismember(obj.resampling_scheme, [2, 3]) % MPF or AMPF
                if isfield(Params, 'state_distance_coefficients')
                    obj.resampling_params.state_distance_coefficients = ...
                        Params.state_distance_coefficients;
                else
                    obj.resampling_params.state_distance_coefficients = [30, 1, 100];
                end
                if isfield(Params, 'cluster_merging_thr')
                    obj.resampling_params.cluster_merging_thr = Params.cluster_merging_thr;
                else
                    obj.resampling_params.cluster_merging_thr = 20;
                end
                if isfield(Params, 'cluster_splitting_thr')
                    obj.resampling_params.cluster_splitting_thr = Params.cluster_splitting_thr;
                else
                    obj.resampling_params.cluster_splitting_thr = 30;
                end
                if isfield(Params, 'n_initial_clusters')
                    obj.resampling_params.n_initial_clusters = Params.n_initial_clusters;
                else
                    obj.resampling_params.n_initial_clusters = 16 * State_space_params.n_patterns;
                end
                if isfield(Params, 'n_max_clusters')
                    obj.resampling_params.n_max_clusters = Params.n_max_clusters;
                else
                    obj.resampling_params.n_max_clusters = 3 * obj.resampling_params.n_initial_clusters;
                end
                if isfield(Params, 'res_int')
                    obj.resampling_params.resampling_interval = Params.res_int;
                else
                    obj.resampling_params.resampling_interval = 30;
                end
            else
                if isfield(Params, 'ratio_Neff')
                    obj.ratio_Neff = Params.ratio_Neff;
                else
                    obj.ratio_Neff = 0.1;
                end
            end
            obj.max_bar_cells = max(Params.whole_note_div * bar_durations);
            obj.use_silence_state = Params.use_silence_state;
        end
        
    end
    
    methods (Static)
        
        outIndex = resampleSystematic( w, n_samples );
        
        outIndex = resampleSystematicInGroups( w, n_samples );
        
        function [groups] = divide_into_fixed_cells(states, state_dims, nCells)
            % divide space into fixed cells with equal number of grid
            % points for position and tempo states
            % states: [n_particles x nStates]
            % state_dim: [nStates x 1]
            groups = zeros(size(states, 1), 1);
            n_r_bins = state_dims(3);
            n_n_bins = floor(sqrt(nCells/n_r_bins));
            n_m_bins = floor(nCells / (n_n_bins * n_r_bins));
            
            m_edges = linspace(1, state_dims(1) + 1, n_m_bins + 1);
            n_edges = linspace(0, state_dims(2) + 1, n_n_bins + 1);
            for iR=1:state_dims(3)
                % get all particles that belong to pattern iR
                ind = find(states(:, 3) == iR);
                [~, BIN_m] = histc(states(ind, 1), m_edges);
                [~, BIN_n] = histc(states(ind, 2), n_edges);
                for m = 1:n_m_bins
                    for n=1:n_n_bins
                        ind2 = intersect(ind(BIN_m==m), ind(BIN_n==n));
                        groups(ind2) = sub2ind([n_m_bins, n_n_bins, state_dims(3)], m, n, iR);
                    end
                end
            end
            if sum(groups==0) > 0
                error('group assignment failed\n')
            end
        end
        
        [outIndex, outWeights, groups] = resample_in_groups(groups, weights, n_max_clusters, warp_fun);
        
        [y, L] = normalizeLogspace(x);
        
        r = logsumexp(X, dim);
    end
    
    
    
end
