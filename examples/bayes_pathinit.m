%script that is called by the ISMIR2016 scripts to initialize the local
%paths correctly
% Andre Holzapfel, 2016
if strcmp(dataset,'Carnatic')
    audio_path = '/home/hannover/Documents/databases/CMCMDa_v2';
    in_file = fullfile(audio_path, 'CarnaticsFromThomasTestList.lab');%fullfile(audio_path, 'guitar_duple.flac');
    out_folder = '/home/hannover/Documents/experiments/results/11_2015_DBNonCNN/CMCMDa_v2/CNNasProbObs_02_12_15';
    dpath = [audio_path filesep];
    talaID = [10:13];
    talaName = {'adi', 'rupaka', 'mChapu', 'kChapu'};
elseif strcmp(dataset,'Ballroom')
    audio_path = '/home/hannover/Documents/databases/Ballroom';
    in_file = fullfile(audio_path, 'test_wpath.lab');%fullfile(audio_path, 'guitar_duple.flac');
    out_folder = '/home/hannover/Documents/experiments/results/11_2015_DBNonCNN/Ballroom/CNNasProbObs_01_31_16';
    dpath = [audio_path filesep];
    talaID = [11:18];
    talaName = {'ChaCha4', 'Jive', 'Quick', 'Rumba' , 'Samba' , 'Tango' , 'Vienna' , 'SlowWaltz'};
else%Hindustani
    audio_path = '/home/hannover/Documents/databases/Hindustani_10_2015/HMDf';
    in_file = fullfile(audio_path, 'HindustaniFromThomasTestList.lab');%fullfile(audio_path, 'guitar_duple.flac');
    out_folder = '/home/hannover/Documents/experiments/results/11_2015_DBNonCNN/HMDf/CNNasProbObs_02_12_15';
    dpath = [audio_path filesep];
    talaID = [20:23];
    talaName = {'teen', 'ek', 'jhap', 'rupak'};
end
addpath('/home/hannover/Documents/experiments/repository/matlabTools/beat-evaluation_fromFlorian/');
if remote == 1
    if strcmp(dataset,'Ballroom')
        bpath = ['/home/hannover/Documents/databases/RemoteJobimDaten/ballroom/data/predictions/_tpool1_log2k_fps100_melspect_501_5_dense512_beatclassnrm_pickopt_model3'];% '
    else
        bpath = ['/home/hannover/Documents/databases/RemoteJobimDaten/andre-hindustani/data/predictions/_tpool1_log2k_fps100_melspect_501_5_dense512_beatclassnrm_pickopt_model2'];% '
    end
elseif remote == 2
    bpath = ['/home/hannover/Documents/databases/RemoteDragonetti/Documents/experiments/results/11_2015_DBNonCNN/' dataset];% '/Tracking/'];
else
    bpath = ['/home/hannover/Documents/experiments/results/11_2015_DBNonCNN/' dataset];% '/Tracking/'];
end