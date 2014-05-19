% Use the beat tracker with a predefined (and trained) model

% Specify a simulation id. The script will create a subirectory "sim_id" in 
% the results folder to save the results
sim_id = 99;
% load config file
Params = ex1_config_bt;
% Create beat tracker object
BT = BeatTracker(Params, sim_id);
BT.init_test_data();
for i_file=1:length(BT.test_data.file_list)
%     BT.load_features(BT.test_data.file_list{i_file});
    results = BT.do_inference(i_file);
end
