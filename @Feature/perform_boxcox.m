% load features and perform boxcox
DataPath = '~/diss/data/beats/ballroom/all';
feat_type{1} = 'lo230_superflux.mvavg';
feat_type{2} = 'hi250_superflux.mvavg';

for i_feat_type = 1:length(feat_type)
    values = [];
   % load data
   listing = dir(fullfile(DataPath, 'beat_activations', ['*.', feat_type{i_feat_type}]));
   for i_file = 1:length(listing)
       [detfunc{i_file}, fr] = Feature.read_activations(listing(i_file).name);
   end
   [lambda] = boxcox(cat(1, detfunc{:}), 1);
end
