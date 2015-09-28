function Results = ex2_beat_tracking_with_pretrained_pf(in_file, ...
    out_folder)
% [Results] = ex2_beat_tracking_with_pretrained_pf(in_file, ...
%   out_folder, model_file)
%   Use the beat tracker with a predefined (and trained) model.
% ----------------------------------------------------------------------
% INPUT Parameter:
%   in_file             : audio file
%   out_folder          : folder, where output (beats and model) are stored
%
% OUTPUT Parameter:
%   Results             : structure array with beat tracking results
%
% 30.07.2015 by Florian Krebs
% ----------------------------------------------------------------------
% Get path of function
[func_path, ~, ~] = fileparts(mfilename('fullpath'));
% pre-trained hmm model
Params.model_fln = fullfile(func_path, 'models/pf_boeck.mat');
if ~exist(Params.model_fln, 'file')
   error('Did not find model file: %s\n', Params.model_fln);
end
Params.inferenceMethod = 'PF';
Params.testLab = in_file;
Params.results_path = out_folder;
% create beat tracker object
BT = BeatTracker(Params);
% do beat tracking
Results = BT.do_inference(1);
% save results
[~, fname, ~] = fileparts(Params.testLab);
BT.save_results(Results, out_folder, fname);
end