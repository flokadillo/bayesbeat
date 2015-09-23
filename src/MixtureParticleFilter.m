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
        
        function [m, n, r, g, w_log] = resampling(obj, m, n, r, g, w_log, iFrame)
            % compute reampling criterion effective sample size
            if (rem(iFrame, obj.resampling_params.resampling_interval) > 0)
                return
            end
            g = obj.cluster(m(:, iFrame+1), n(:, iFrame), r(:, iFrame+1), g);
            [newIdx, w_log, g] = obj.resample_in_groups(g, w_log, ...
                obj.resampling_params.n_max_clusters);
            m(:, 1:iFrame+1) = m(newIdx, 1:iFrame+1);
            r(:, 1:iFrame+1) = r(newIdx, 1:iFrame+1);
            n(:, 1:iFrame) = n(newIdx, 1:iFrame);
        end
        
        function [groups] = cluster(obj, m, n, r, groups_old)
            % states: [n_particles x nStates]
            % state_dim: [nStates x 1]
            % groups_old: [n_particles x 1] group labels of the particles
            %               after last resampling step (used for initialisation)
            warning('off');
            [group_ids, ~, new_group_idx] = unique(groups_old);
            k = length(group_ids); % number of unique clusters
            % adjust the range of each state variable to make equally
            % important for the clustering
            dim_weighting = obj.resampling_params.state_distance_coefficients;
            bar_durations = obj.state_space.meter_from_pattern(1, :) ./ ...
                obj.state_space.meter_from_pattern(2, :);
            points = zeros(obj.n_particles, 4);
            points(:, 1) = (sin(m * 2 * pi ./ ...
                obj.state_space.max_position_from_pattern(r)) + 1) * ...
                dim_weighting(1) .* bar_durations(r);
            points(:, 2) = (cos(m * 2 * pi ./ ...
                obj.state_space.max_position_from_pattern(r)) + 1) * ...
                dim_weighting(1) .* bar_durations(r);
            points(:, 3) = n * dim_weighting(2);
            points(:, 4) =(r-1) * dim_weighting(3) + 1;
            % compute centroid of clusters
            centroids = zeros(k, 4);
            for i_dim=1:size(points, 2)
                centroids(:, i_dim) = accumarray(new_group_idx, ...
                    points(:, i_dim), [], @mean);
            end
            % do k-means clustering
            options = statset('MaxIter', 1);
            [groups, centroids, total_dist_per_cluster] = kmeans(points, k, ...
                'replicates', 1, 'start', centroids, 'emptyaction', 'drop', ...
                'Distance', 'sqEuclidean', 'options', options);
            % remove empty clusters
            total_dist_per_cluster = total_dist_per_cluster(...
                ~isnan(centroids(:, 1)));
            centroids = centroids(~isnan(centroids(:, 1)), :);
            [group_ids, ~, j] = unique(groups);
            group_ids = 1:length(group_ids);
            groups = group_ids(j)';
            % check if centroids are too close
            merging = 1;
            merged = 0;
            while merging && (size(centroids, 1) > 1)
                % compute distance from each centroid to each other
                D = squareform(pdist(centroids, 'euclidean'), 'tomatrix');
                % extract lower triangular part
                ind = (tril(D, 0) > 0);
                D(ind) = nan;
                D(logical(eye(size(centroids, 1)))) = nan;
                % find the two closest clusters
                [dist_closest, idx_closest] = min(D(:));
                if dist_closest(1) > obj.resampling_params.cluster_merging_thr,
                    % no merging has to be performed
                    merging = 0;
                else
                    [c1, c2] = ind2sub(size(D), idx_closest(1));
                    % rename group c2 to c1
                    groups(groups==c2) = c1;
                    % rename group ids
                    groups(groups>=c2) = groups(groups>=c2) - 1;
                    % update centroid of c1 after merging
                    centroids(c1, :) = mean(points(groups==c1, :));
                    % update total (squared) dsitance of c1
                    total_dist_per_cluster(c1) = sum(sum(bsxfun(@minus, ...
                        points(groups==c1, :), centroids(c1, :)).^2));
                    % delete cluster c2
                    centroids = centroids([1:c2-1, c2+1:end], :);
                    total_dist_per_cluster = ...
                        total_dist_per_cluster([1:c2-1, c2+1:end]);
                    merged = 1;
                end
            end
            % check if cluster spread is too high
            split = 0;
            if merged
                [group_ids, ~, j] = unique(groups);
                group_ids = 1:length(group_ids);
                groups = group_ids(j)';
            end
            n_parts_per_cluster = hist(groups, group_ids);
            to_split_idx = find((total_dist_per_cluster ./ ...
                n_parts_per_cluster') > ...
                obj.resampling_params.cluster_splitting_thr);
            for iCluster = 1:length(to_split_idx)
                % find particles that belong to the cluster to split
                parts_idx = find((groups == to_split_idx(iCluster)));
                % put second half into a new group (the actual clustering takes
                % place in the refinement kmeans)
                groups(parts_idx(round(length(parts_idx)/2)+1:end)) = ...
                    max(groups) + 1;
                % update centroid
                centroids(to_split_idx(iCluster), :) = ...
                    mean(points(parts_idx(1:round(length(parts_idx)/2)), :), 1);
                % add new centroid
                centroids = [centroids; mean(points(parts_idx(round(...
                    length(parts_idx)/2)+1:end), :), 1)];
                split = 1;
            end
            if split || merged
                [groups, ~, ~] = kmeans(points, [], 'replicates', 1, ...
                    'start', centroids, 'emptyaction', 'drop', ...
                    'Distance', 'sqEuclidean', 'options', options);
                [group_ids, ~, j] = unique(groups);
                group_ids = 1:length(group_ids);
                groups = group_ids(j)';
            end
            warning('on');
        end
        
        
        
    end
    
    methods (Static)
        function [outIndex, outWeights, groups_new] = resample_in_groups(...
                groups, w_log, n_max_clusters, warp_fun)
            %  [outIndex] = resample_in_groups(groups, w_log)
            %  resample particles in groups separately
            % ----------------------------------------------------------------------
            %INPUT parameter:
            % groups          : filename (e.g., Media-105907(0.0-10.0).beats)
            % weight          : logarithmic weights
            %
            %
            %OUTPUT parameter:
            % outIndex        : new resampled indices
            %
            % 25.09.2012 by Florian Krebs
            % ----------------------------------------------------------------------
            if all(isnan(w_log))
               error('All weights are zero. Aborting...\n'); 
            end
            w_log = w_log(:);
            groups = groups(:);
            % group weights according to groups
            w_per_group = accumarray(groups, w_log, [], @(x) {x});
            % sum weights of each group in the log domain
            tot_w = cellfun(@(x) ParticleFilter.logsumexp(x, 1), w_per_group);
            % check for groups with zero weights (log(w)=-inf) and remove those
            if sum(isnan(tot_w)) > 0
                bad_groups = find(isnan(tot_w));
            else
                bad_groups = [];
            end
            % kill clusters with lowest weight to prevent more than n_max_clusters
            % clusters
            if length(tot_w) - length(bad_groups) > n_max_clusters
                [~, groups_sorted] = sort(tot_w, 'descend');
                fprintf('    too many groups (%i)! -> removing %i\n', ...
                    length(tot_w) - length(bad_groups), ...
                    length(tot_w) - length(bad_groups) - n_max_clusters);
                bad_groups = unique([bad_groups; ...
                    groups_sorted(n_max_clusters+1:end)]);
            end
            
            %determine indices of particles for each cluster
            id_per_group = accumarray(groups, (1:length(w_log))', [], @(x) {x});
            id_per_group(bad_groups) = [];
            id_per_group = cell2mat(id_per_group);
            
            n_groups = length(tot_w) - length(bad_groups);
            % cumulative sum of particles per group. Each group should have an
            % approximative equal number of particles.
            %targeted number of particles per cluster
            parts_per_group = diff(round(linspace(0, length(w_log), n_groups+1)));
            parts_per_group(end) = length(w_log) - sum(parts_per_group(1:end-1));
            w_norm = exp(w_log - tot_w(groups));
            if exist('warp_fun', 'var')
                % do warping
                w_warped = warp_fun(w_norm);
                % normalize weights before resampling
                sum_warped_per_group = accumarray(groups, w_warped);
                w_warped_norm = w_warped ./ sum_warped_per_group(groups);
                w_warped_per_group = accumarray(groups, w_warped_norm, [], ...
                    @(x) {x});
                % resample
                w_warped_per_group(bad_groups) = [];
                outIndex = MixtureParticleFilter.resample_systematic_in_groups( ...
                    w_warped_per_group, ...
                    parts_per_group);
                outIndex = id_per_group(outIndex);
                groups_new = groups(outIndex);
                % do unwarping
                w_fac = w_norm ./ w_warped;
                norm_fac = accumarray(groups_new, w_fac(outIndex));
                outWeights = log(w_fac(outIndex)) + tot_w(groups_new) - ...
                    log(norm_fac(groups_new));
            else
                w_norm_per_group = accumarray(groups, w_norm, [], @(x) {x});
                w_norm_per_group(bad_groups) = [];
                outIndex = MixtureParticleFilter.resample_systematic_in_groups(...
                    w_norm_per_group, ...
                    parts_per_group);
                outIndex = id_per_group(outIndex);
                groups_new = groups(outIndex);
                % divide total weight among new particles
                outWeights = tot_w(groups_new) - log(mean(parts_per_group));
            end
        end
        
        function [ indx ] = resample_systematic_in_groups( W, n_samples )
            % W ... cell array of normalised weights [n_groups x 1]
            % n_samples ... [1 x n_cluster] number of samples after resampling
            Q = cumsum(cell2mat(W));
            n_particles = sum(n_samples);
            T = linspace(0, length(n_samples)-length(n_samples)/n_particles, n_particles) ...
                + rand(1)*length(n_samples)/n_particles;
            T(n_particles+1) = length(n_samples);
            [~, indx] = histc(T, [0; Q]);
            indx = indx(1:end-1);
        end
        
    end
    
end

