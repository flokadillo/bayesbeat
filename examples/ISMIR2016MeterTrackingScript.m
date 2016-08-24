% Script to run the meter tracking model from "Bayesian meter tracking on learned signal representations" by Andre Holzapfel and Thomas Grill.
% works so far only for using the HMM 2015 model, not implemented so far
% for other inferences.
% Major changes to the repository are:
% 1. the RNN_db observation model, that
% allows to load beat and downbeat activations.
% 2. the focussing of the state space to tempo peaks. This is temporarily
% done by "misusing" the min_tempo_bpm variable to store the observed tempo
% peaks, instead of only one tempo value for the mean tempo of the state
% space. In a final version, this should be done using a separate variable
% in order to avoid conflicts.
% In order to work, the network activations need to be stored in the audio
% path in a subfolder called "beat_activations". The activation files
% should contain three lines: 
% FRAMERATE: 100.0000 %framerate in Hz
% DIMENSION: 3179 %length of the following activation vector
% third row: activation values
%
% Andre Holzapfel, 2016
dataset = 'Ballroom';
remote = 0;%this is just a variable that was used to run experiments on remote machines. Set to zero!
%switch on/off the focussing of the tempo states around peaks in the
%autocorrelation function of the input features. This was found to speed up
%inference especially for rhythmically simple input.
focusTempoStates = 1;
%how many tempo candidates should be picked? The strategy is to pick the
%maximum of the ACF, and then choose NumTempoCands-1 peaks in the tempo octaves above and below
%this maximum.
NumTempoCands = 5;
%initialize local paths for the three used datasets.
bayes_pathinit;
%read the list of test files
fid = fopen(in_file, 'r');
file_list = textscan(fid, '%s', 'delimiter', '\n');
file_list = file_list{1};
fclose(fid);
%do piece wise evaluation, using a tempo estimation from its activations
for i = 1:length(file_list)
    [~,fname,~] = fileparts(file_list{i});
    fprintf('Processing file %i of %i: %s\n ',i,length(file_list),fname);
    %read the activations
    fln = fullfile(audio_path,'audio/beat_activations',[fname '.beat_cnn_activations']);
    fid = fopen(fln,'r');
    c = textscan(fid, '%s %f', 1);
    if strcmpi(c{1},'framerate:')
        fr = c{2};
    else
        fprintf('Warning Feature.read_activations: No FRAMERATE field found\n');
        fr = [];
    end
    c = textscan(fid, '%s %d', 1);
    if ~strcmp(c{1}, 'DIMENSION:')
        % file has no header 'DIMENSION'
        frewind(fid);
        % ignore first line
        c = textscan(fid, '%s %f', 1);
    end
    act = textscan(fid, '%f');
    act = act{1};
    fclose(fid);
    %analyze the autocorrelation of the beat activations
    maxdur = 12;
    maxlag = maxdur*fr;
    AC = xcorr(act,maxlag,'coeff');
    AC = AC(ceil(length(AC)/2)+1:end);
    AC_tempi = 60./[1/fr:1/fr:maxdur]';%tempo values of the ACF
    AC = AC(AC_tempi<500);%keep only those tempi below 500 bpm
    AC_tempi = AC_tempi(AC_tempi<500);
    [val,ind]= max(AC);%determine highest peak > tempo candidate
    min_tempo_bpm = 0.4*AC_tempi(ind);%allow range of (slightly more than) two octaves around that
    if min_tempo_bpm<10
        min_tempo_bpm = 10;
    end
    max_tempo_bpm = 2.2*AC_tempi(ind);
    %until here, tempo range is only restricted to the two octave range
    %around the maximum ACF peak. If the focusing of the tempo states around several peak values is desired, then
    %these peak values need to be added to the min_tempo_bpm variable.
    if focusTempoStates == 1
        %following part tries to pick up the tempo candidates
        [maxtab, mintab] = peakdet(AC,0.001);
        [sortedACpeakVals,sortedACpeakInd]  = sort(maxtab(:,2),'descend');
        tempocands = AC_tempi(maxtab(sortedACpeakInd,1));
        AC_vals = AC(maxtab(sortedACpeakInd,1));
        inRange = min_tempo_bpm<tempocands & max_tempo_bpm>tempocands & AC_vals>mean(AC);
        min_tempo_bpm = tempocands(inRange);%tempo cands
        if length(min_tempo_bpm) > NumTempoCands
            min_tempo_bpm = min_tempo_bpm(1:NumTempoCands);
        end
    end
    ISMIR2016Tracking(file_list{i}, out_folder,min_tempo_bpm,max_tempo_bpm);%,transition_lambda,tempo_states);
end