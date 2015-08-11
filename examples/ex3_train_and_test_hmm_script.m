% Script to run examples 3, train an HMM and test it.
audio_path = fullfile(pwd, 'data', 'audio');
in_file = fullfile(audio_path, 'guitar_duple.flac');
train_files = {'guitar_duple.flac', 'guitar_triple.flac'};
train_files = cellfun(@(x) fullfile(audio_path, x), train_files, ...
    'UniformOutput', 0);
out_folder = fullfile(pwd, 'out');
Results = ex3_train_and_test_hmm(in_file, train_files, out_folder);