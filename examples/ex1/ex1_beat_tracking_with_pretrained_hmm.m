% Use the beat tracker with a predefined (and trained) model. Please make 
% sure to add the path to your local copy of the
% bayes_beat package in the config file ex1_config_bt.m.

% specify a simulation id. The script will create a subirectory "sim_id" in 
% the results folder to save the results
sim_id = 1;
% load config file
Params = ex1_config_bt;
% create beat tracker object
BT = BeatTracker(Params, sim_id);
% set up test data
BT.init_test_data();
% do beat tracking
results = BT.do_inference(1);
% save results
[~, fname, ~] = fileparts(Params.testLab);
BT.save_results(results, fullfile(Params.results_path, num2str(sim_id)), fname);