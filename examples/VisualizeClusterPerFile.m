%small script that reads the files in a list and initializes the models for
%each individual file. Can be used e.g. to plot the feature data for a
%specific file.
%Andre Holzapfel, 2016
dataset = 'Hind';
remote = 0;%check
bayes_pathinit;
in_file = '/home/hannover/Documents/databases/Hindustani_10_2015/HMDf/HindustaniFromThomasTestListTaala1.lab';
fid = fopen(in_file, 'r');
file_list = textscan(fid, '%s', 'delimiter', '\n');
file_list = file_list{1};
Params.transition_model_type = '2015';
% Use kmeans clustering with two clusters
Params.cluster_type = 'kmeans';
Params.n_clusters = 1;%in ISMIR 2014 paper this value was set to 2.
Params.pattern_size = 'bar';
Params.testLab = in_file;
Params.data_path = out_folder;
Params.results_path = out_folder;
Params.feat_type = {'downbeat_cnn_activations', 'beat_cnn_activations'};
Params.frame_length = 1/100;
%do piece wise evaluation, using a tempo estimation from its activations
for i = 1:length(file_list)
    Params.trainLab = file_list{i};
    [~,fname,~] = fileparts(file_list{i});
    annpath = [audio_path '/annotations/beats/'];
    annFile = [fname '.beats'];
    ann = load([annpath annFile],'-ascii');
    beats = ann(:,1);
    temp = sort(diff(beats));
    lenn = length(beats);
    tempo = 60./median(temp(round(lenn/10):round(0.9*lenn)))
    % create beat tracker object
    BT = BeatTracker(Params);
end