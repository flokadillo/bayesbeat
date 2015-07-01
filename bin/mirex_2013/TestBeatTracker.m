% TestBeatTracker

% computes beats of all files in directory DataPath

% needs:
%   compute_beats_mirex_2013

DataPath = '~/diss/data/beats/collins_aes_2013/';
OutPath = '~/diss/MIREX/2013/submissions/FK1_2013/temp/';

%% Compute BEATS
listing = dir(fullfile(DataPath,'*.wav'));
fprintf('\nComputing beats ...\n');
for g=1:length(listing)
    fprintf('%i/%i\n',g,length(listing));
    if isempty(listing(g).name)
        continue;
    end
    [~, fname, ~] = fileparts(listing(g).name);
    inputfln = fullfile(DataPath, [fname,'.wav']);
    output_file_name = fullfile(OutPath, [fname, '.beats.txt.new']);
    compute_beats_mirex_2013( inputfln, output_file_name );
end