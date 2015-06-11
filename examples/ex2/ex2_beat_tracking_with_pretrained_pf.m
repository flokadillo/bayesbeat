% Use the beat tracker with a predefined (and trained) particle filter
% model. Please edit the following paths:
% 1) bayes beat base folder
path_to_bayes_beat = '~/diss/src/matlab/beat_tracking/bayes_beat';
% 2) audio file to be analysed (flac or wav supported):
audio_test = '~/diss/src/matlab/beat_tracking/bayes_beat/examples/audio/train1.flac';
% ------------------------------------------------------------------------
% specify a simulation id. The script will create a subirectory "sim_id" in 
% the results folder to save the results
sim_id = 1;
% load config file
Params = ex2_config_bt(path_to_bayes_beat);
Params.testLab = audio_test;
% create beat tracker object
BT = BeatTracker(Params, sim_id);
% initialise model
BT.init_model;
% set up test data
BT.init_test_data();
% do beat tracking
results = BT.do_inference(1);
% save results
[~, fname, ~] = fileparts(Params.testLab);
BT.save_results(results, fullfile(Params.results_path, num2str(sim_id)), fname);