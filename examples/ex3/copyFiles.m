clear
close all
clc
addpath('E:\UPFWork\PhD\Code\githubCode\ProcessData');
bpath = 'E:\UPFWork\PhD\Data\CMCMDa_v2\';
opath = 'E:\UPFWork\PhD\Code\BayesBeat\data\CMCMDa_v2\';
ann = 'annotations\';
audio = 'audio\';
beats = [ann 'beats\'];
meter = [ann 'meter\'];
aftype = 'mp3';
afext = ['.' aftype];
if ~(isdir([opath audio]))
    mkdir([opath audio]);
end
if ~(isdir([opath ann]))
    mkdir([opath ann]);
end
if ~(isdir([opath beats]))
    mkdir([opath beats]);
end
if ~(isdir([opath meter]))
    mkdir([opath meter]);
end

fp = fopen([bpath 'filelist.txt'],'rt');
AA = textscan(fp,'%s\n');
list = AA{1};
fclose(fp);
talas = {'adi', 'rupakam', 'mishraChapu', 'khandaChapu'};
ts = {'8/4', '3/4', '7/8', '5/8'};
for k = 1:length(list)
    [tala, fname, ~] = fileparts(list{k});
    tInd(k) = find(ismember(talas, tala));
    ipwav = [bpath aftype filesep list{k} afext];
    opwav = [opath audio fname afext];
    % sysStr = strcat({'lame --decode '}, ipwav, {' '}, opwav);
    % system(sysStr{1});
    copyfile(ipwav,opwav);
    ipbts = [bpath 'annot\' list{k} '.csv'];
    ipann = dlmread(ipbts);
    opann = interpolateAnn(ipann,tInd(k));
    if aftype == 'mp3'
        opann(:,1) = opann(:,1) - 2260/44100;       % Hack due to mp3 files
    end
    opbts = [opath beats fname '.beats'];
    dlmwrite(opbts,opann,'precision','%10.6f');
    opmeter = [opath meter fname '.meter'];
    fp = fopen(opmeter,'wt');
    fnames{k} = fname;
    fprintf(fp, '%s', ts{tInd(k)});
    fclose(fp);
    k
end

trainInd = [];
testInd = [];
for k = 1:length(unique(tInd))
    pts = find(tInd == k);
    pts = pts(randperm(length(pts)));
    trainInd = [trainInd pts(1:length(pts)/2)];
    testInd = [testInd pts(length(pts)/2+1:end)];
end
list1 = fnames(trainInd);
list2 = fnames(testInd);
fp1 = fopen(['train_1' '.lab'],'wt');
fp2 = fopen(['test_2' '.lab'],'wt');
for p = 1:length(list1)
    fprintf(fp1, '%s\n',[opath audio list1{p} '.wav']);
    fprintf(fp2, '%s\n',[opath audio list1{p} '.wav']);
end
fclose(fp1);
fclose(fp2);
    
fp1 = fopen(['train_2.lab'],'wt');
fp2 = fopen(['test_1.lab'],'wt');
for p = 1:length(list1)
    fprintf(fp1, '%s\n',[opath audio list2{p} '.wav']);
    fprintf(fp2, '%s\n',[opath audio list2{p} '.wav']);
end
fclose(fp1);
fclose(fp2);


