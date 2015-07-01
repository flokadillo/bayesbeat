% Train a HMM model from data.  Please edit the following paths:
% 1) bayes beat base folder
path_to_bayes_beat = '../../';
% 2) audio file to be analysed (flac or wav supported):
audio_test = '~/diss/src/matlab/beat_tracking/bayes_beat/examples/audio/train1.flac';

% Load the parameters
Params = ex3_config_bt(path_to_bayes_beat);
% specify a simulation id
sim_id = 1;

% CLUSTERING THE DATASET
Clustering = RhythmCluster('examples/ex3/test_3_4.lab', Params.feat_type, ...
    Params.frame_length, Params.data_path, 'bar');
% cluster the dataset according to the meter of each file
Params.clusterIdFln = Clustering.make_cluster_assignment_file('meter');

% TRAINING THE MODEL

% create beat tracker object
BT = BeatTracker(Params, sim_id);
% set up test_data
BT.init_test_data();
% train model
BT.train_model();

% TEST THE MODEL
% do beat tracking
results = BT.do_inference(1);
% save results
[~, fname, ~] = fileparts(Params.testLab);
BT.save_results(results, fullfile(Params.results_path, num2str(sim_id)), fname);
