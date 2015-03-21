clear
close all
clc
addpath('../../../CommonPoolCodeGeneral/Davies_beat_error_histogram/');
bpath = 'E:\UPFWork\PhD\Code\BayesBeat\results\';
annpath = 'E:\UPFWork\PhD\Code\BayesBeat\data\CMCMDa_small\annotations\beats\';
exptName = 'CMCMDa_small\MPF_6000_run2\';
exppath = [bpath exptName filesep];
oFiles = dir([exppath,'*.beats.txt']);
talaID = [10 11 12 13];
talaName = {'adi', 'rupakam', 'mishraChapu', 'khandaChapu'};
for k = 1:length(oFiles)
    annout = dlmread([exppath oFiles(k).name]);
    annout(:,2) = round(10*(annout(:,2) - floor(annout(:,2))));
    [~, annFile, ~] = fileparts(oFiles(k).name);
    talaIndicator(k) = find(ismember(talaID, str2num(annFile(4:5))));
    fileTalaName{k} = talaName(talaIndicator(k));
    fileID(k) = str2num(annFile(4:8));
    ann = load([annpath annFile],'-ascii');
    opSamas = annout((annout(:,2) == 1),1);
    opBeats = annout(:,1);
    samas = ann(ann(:,2)==1,1);
    beats = ann(:,1);
    % Sama metrics
    res(k).sama.pScore = be_pScore(samas,opSamas);
    [res(k).sama.fMeas res(k).sama.precision res(k).sama.recall res(k).sama.Ameas]...
        = be_fMeasure(samas,opSamas);
    res(k).sama.infoGain = be_informationGain(samas,opSamas);
    [res(k).sama.cmlC res(k).sama.cmlT res(k).sama.amlC res(k).sama.amlT]...
        = be_continuityBased(samas,opSamas);
    % Beat metrics
    res(k).beat.pScore = be_pScore(beats,opBeats);
    [res(k).beat.fMeas res(k).beat.precision res(k).beat.recall res(k).beat.Ameas]...
        = be_fMeasure(beats,opBeats);
    res(k).beat.infoGain = be_informationGain(beats,opBeats);
    [res(k).beat.cmlC res(k).beat.cmlT res(k).beat.amlC res(k).beat.amlT]...
        = be_continuityBased(beats,opBeats);
    fprintf('Processing file... %s\n',oFiles(k).name);
end

sm = [res.sama];
s.pScore = [sm.pScore];
s.fMeas = [sm.fMeas];
s.precision = [sm.precision];
s.recall = [sm.recall];
s.infoGain = [sm.infoGain];
s.cmlC = [sm.cmlC];
s.cmlT = [sm.cmlT];
s.amlC = [sm.amlC];
s.amlT = [sm.amlT];
    
bt = [res.beat];
b.pScore = [bt.pScore];
b.fMeas = [bt.fMeas];
b.precision = [bt.precision];
b.recall = [bt.recall];
b.infoGain = [bt.infoGain];
b.cmlC = [bt.cmlC];
b.cmlT = [bt.cmlT];
b.amlC = [bt.amlC];
b.amlT = [bt.amlT];
sVec = [mean(s.pScore) mean(s.fMeas) mean(s.precision) mean(s.recall) mean(s.infoGain)...
        mean(s.cmlC) mean(s.cmlT) mean(s.amlC) mean(s.amlT)]
bVec = [mean(b.pScore) mean(b.fMeas) mean(b.precision)...
    mean(b.recall) mean(b.infoGain) mean(b.cmlC) mean(b.cmlT)...
    mean(b.amlC) mean(b.amlT)]
dlmwrite([exppath 'opResults.txt'], sVec, '-append');
dlmwrite([exppath 'opResults.txt'], bVec, '-append');
clear s b
for k = 1:length(unique(talaIndicator))
    chosen = find(talaIndicator == k);
    sm = [res(chosen).sama];
    s.pScore = [sm.pScore];
    s.fMeas = [sm.fMeas];
    s.precision = [sm.precision];
    s.recall = [sm.recall];
    s.infoGain = [sm.infoGain];
    s.cmlC = [sm.cmlC];
    s.cmlT = [sm.cmlT];
    s.amlC = [sm.amlC];
    s.amlT = [sm.amlT];
    
    bt = [res(chosen).beat];
    b.pScore = [bt.pScore];
    b.fMeas = [bt.fMeas];
    b.precision = [bt.precision];
    b.recall = [bt.recall];
    b.infoGain = [bt.infoGain];
    b.cmlC = [bt.cmlC];
    b.cmlT = [bt.cmlT];
    b.amlC = [bt.amlC];
    b.amlT = [bt.amlT];
    
    fprintf('Taala: %s\n',talaName{k});
    disp('Sama accuracies')
    sVec = [mean(s.pScore) mean(s.fMeas) mean(s.precision) mean(s.recall) mean(s.infoGain)...
        mean(s.cmlC) mean(s.cmlT) mean(s.amlC) mean(s.amlT)]
    disp('Beat accuracies')
    bVec = [mean(b.pScore) mean(b.fMeas) mean(b.precision)...
    mean(b.recall) mean(b.infoGain) mean(b.cmlC) mean(b.cmlT)...
    mean(b.amlC) mean(b.amlT)]
    perf(k).sama = s;
    perf(k).beat = b;
    clear b s sm bt
    dlmwrite([exppath 'opResults.txt'], sVec, '-append');
    dlmwrite([exppath 'opResults.txt'], bVec, '-append');
end
