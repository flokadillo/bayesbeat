function Params = fit_distribution(obj, data)
% Params = fit_distribution(obj, dataPerClusterAndPosition)
% Fit a given distribution to the data and return the learned parameters.
% ------------------------------------------------------------------------
%INPUT parameters:
% data      cell array of features [n_files, n_patterns, n_position_cells,
%               feat_dim]
%OUTPUT parameters:
% Params    structure array of learned parameters [n_patterns, n_position_cells]
%
% 27.08.2015 by Florian Krebs
% ------------------------------------------------------------------------
warning('off');
[n_files, n_patterns, n_position_cells, ~] = size(data);
Params = cell(n_patterns, n_position_cells);
% Set options
switch obj.dist_type
    case 'MOG'
        options = statset('MaxIter', 200);
        n_replicates = 10;
%         fprintf('WARNING: Don''t forget to reset <n_replicates>\n');
end
for i_pattern=1:n_patterns
    for i_pos=1:n_position_cells
        if n_files == 1
            % resulting featureValues should be a matrix [nValues x featDim]
            % if files are squeezed out we have to transpose
            feature_values = cell2mat(squeeze(...
                data(:, i_pattern, i_pos, :))');
        else
            feature_values = cell2mat(squeeze(...
                data(:, i_pattern, i_pos, :)));
        end
        if isempty(feature_values),
            break;
        end
        switch obj.dist_type
            case 'MOG'
                Params{i_pattern, i_pos} = gmdistribution.fit(feature_values, ...
                    2, 'Options', options, 'Regularize', 1e-10, ...
                    'Replicates', n_replicates);
            otherwise
                error('distribution type %s unknown !', obj.dist_type);
        end
    end
end
warning('on');
end
