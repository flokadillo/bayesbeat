function Params = fit_distribution(obj, data)
% Params = fit_distribution(obj, dataPerClusterAndPosition)
% Fit a given distribution to the data and return the learned parameters.
% ------------------------------------------------------------------------
%INPUT parameters:
% data      cell array of features [n_patterns, n_position_cells,
%               feat_dim]
%OUTPUT parameters:
% Params    structure array of learned parameters [n_patterns, n_position_cells]
%
% 27.08.2015 by Florian Krebs
% changed 30.11.2015 by Andre Holzapfel 
% ------------------------------------------------------------------------
warning('off');
[n_patterns, n_position_cells, ~] = size(data);
Params = cell(n_patterns, n_position_cells);
% Set options
switch obj.dist_type
    case 'MOG'
        options = statset('MaxIter', 200);
        n_replicates = 5;
        n_mix_components = 2;
end
for i_pattern=1:n_patterns
    for i_pos=1:n_position_cells
        feature_values = cell2mat(squeeze(data(i_pattern, i_pos, :))');
        if isempty(feature_values),
            break;
        end
        switch obj.dist_type
            case 'MOG'
                Params{i_pattern, i_pos} = gmdistribution.fit(feature_values, ...
                    n_mix_components, 'Options', options, 'Regularize', 1e-10, ...
                    'Replicates', n_replicates);
            otherwise
                error('distribution type %s unknown !', obj.dist_type);
        end
    end
end
warning('on');
end
