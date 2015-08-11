function [ ] = compute_beats_mirex_2013( input_file_name, output_file_name )
% [ ] = compute_beats_mirex_2013( input_file_name, output_file_name )
%   Computes beat times of input_file_name and saves them to
%   output_file_name
% ----------------------------------------------------------------------
%INPUT parameter:
% input_file_name   : path and filenam of the input WAV file
% output_file_name  : path and filename of the output TXT file
%
% DEPENDS ON
% bayes_beat class
%
% 30.08.2013 by Florian Krebs
% ----------------------------------------------------------------------
% get path of function
[func_path, ~, ~] = fileparts(mfilename('fullpath'));
fprintf('Processing %s ',input_file_name);
% ---------- SET PARAMETERS --------------------------------------------
Params.model_fln = fullfile(func_path, 'mirex_2013_hmm.mat');
Params.use_mex_viterbi = 1;
Params.testLab = input_file_name;
% ---------- COMPUTE BEATS --------------------------------------------
beat_tracker = BeatTracker(Params);
beat_tracker.init_test_data();
results = beat_tracker.do_inference(1);
beats = results{1}(:,1);
% ---------- SAVE BEATS --------------------------
[pathstr, ~, ~] = fileparts(output_file_name);
if ~exist(pathstr,'dir') && ~isempty(pathstr)
    fprintf('\n''%s'' not found, creating new\n',pathstr);
    mkdir(pathstr);
end
fid = fopen(output_file_name, 'w');
fprintf(fid,'%.3f\n',beats);
fclose(fid);
end


