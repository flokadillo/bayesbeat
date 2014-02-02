function params = fit_distribution(obj, dataPerClusterAndPosition)

fprintf('* Set up observation model .');
if strcmp(obj.dist_type,'histogram') || strcmp(obj.dist_type,'multivariateHistogram'),
    for  m=1:length(ClusterIdx)
        RhytPat.centerbins = (logspace(0, 1, nHistBins)-1) / 10;
        RhytPat.centerbins(end) = 1;
    end
end

%Init = generate_init_params_gmdistribution(dataPerClusterAndPosition);

[~, nPatterns, nPos, ~] = size(dataPerClusterAndPosition);

params = cell(nPatterns, nPos);

for iPattern=1:nPatterns % all clusters
    fprintf('.');
    
    for iPos=1:nPos
        
        featureValues = cell2mat(squeeze(dataPerClusterAndPosition(:, iPattern, iPos, :)));
        
        if isempty(featureValues), break; end
        
        switch obj.dist_type
            case 'gamma'
                PD = fitdist(featureValues, 'gamma');
                params{iPattern, iPos} = PD.Params';
            case 'MOG'
                options = statset('MaxIter', 200);
                params{iPattern, iPos} = gmdistribution.fit(featureValues, 2, 'Options', options, 'Regularize', 1e-10);
            case 'MOG3'
                options = statset('MaxIter', 200);
                params{iPattern, iPos} = gmdistribution.fit(featureValues, 3, 'Options', options, 'Regularize', 1e-10);
            case 'gauss'
                [params{iPattern, iPos}(1,1), params{iPattern, iPos}(2,1)] = ...
                    normfit(featureValues);
            case 'invGauss'
                PD = fitdist(featureValues, 'inversegaussian');
                params{iPattern, iPos} = PD.Params';
            case 'gauss0'
                params{iPattern, iPos}(1,1) = 0;
                params{iPattern, iPos}(2,1) = sqrt(sum(featureValues.^2)/(length(featureValues)-1));
            case 'histogram'
                params{iPattern, iPos} = histc(featureValues, params.centerbins);
                % normalize
                params{iPattern, iPos} = params{iPattern, iPos} / sum(params{iPattern, iPos});
            case 'multivariateHistogram'
                params{iPattern, iPos} = hist3(featureValues{iPattern, iPos}, cell({params.centerbins, params.centerbins}));
                % normalize
                params{iPattern, iPos} = params{iPattern, iPos} ./ sum(params{iPattern, iPos}(:));
            otherwise
                error('distribution type %s unknown !', obj.dist_type);
        end
        
    end
end
fprintf(' done\n');
end
