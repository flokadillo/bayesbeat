clear
close all
clc
% dataset = 'BallroomDataset';
% dataset = 'CMCMDa_small';
% dataset = 'HMDs';
% dataset = 'HMDf';
dataset = 'HMDl';
% dataset = 'CMCMDa_v2';
% dataset = 'CretanLeapingDances';
addpath('../../CommonPoolCodeGeneral/Davies_beat_error_histogram/');
bpath = ['/media/Code/UPFWork/PhD/BayesResultsFull/' dataset '/Tracking/'];
dpath = ['/media/Code/UPFWork/PhD/Data/' dataset filesep];
annpath = [dpath 'annotations/beats/'];
exptName = 'PF_bar_AMPF_Full_NoHop_1500pp';
exppath = [bpath exptName filesep];
% talaID = [10:13];
talaID = [20:23];
% talaID = [11:18];
nExp = 3;
nFold = 2;
numPatts = [1 2 4];
% talaName = {'adi', 'rupaka', 'mChapu', 'kChapu'};
talaName = {'teen', 'ek', 'jhap', 'rupak'};
% talaName = {'cretan'};
% talaName = {'ChaChaCha', 'Jive' , 'Quickstep', 'Rumba' , 'Samba' , 'Tango', 'VienneseWaltz', 'Waltz'};
for t = 1:length(talaName)
    fpp = fopen([dpath 'filelist_' talaName{t} '.txt']);
    filelist = textscan(fpp,'%s');
    filelist = filelist{1};
    fclose(fpp);
    for p = 1:length(numPatts)
        talabasepath = [exppath talaName{t} filesep 'nPatts_' num2str(numPatts(p)) filesep];
        for ex = 1:nExp
%             oFiles = {};
%             for f = 1:nFold
%                 sim_id = 1000*ex+f;
%                 % sim_id = '';
%                 fileList = dir([talabasepath, num2str(sim_id), filesep, '*.beats.txt']);
%                 for k = 1:length(fileList);
%                     oFiles = [oFiles, {fullfile(talabasepath, num2str(sim_id), fileList(k).name)}];
%                 end
%             end
%            length(oFiles)
            % Process each file now
            for k = 1:length(filelist)
                % Read the ground truth first
                annFile = [filelist{k} '.beats'];
                talaIndicator(k) = find(ismember(talaID, str2num(annFile(4:5))));
                fileTalaName{k} = talaName{talaIndicator(k)};
                fileID(k) = str2num(annFile(4:8));
                fileNum(k) = str2num(annFile(1:2));
                ann = load([annpath annFile],'-ascii');
                samas = ann(ann(:,2)==1,1);
                beats = ann(:,1);
                % Tempo GT
                temp = sort(diff(beats));
                lenn = length(beats);
                res(k).bpmGT = 60./median(temp(round(lenn/10):round(0.9*lenn)));
                % Get the output file name next
                gotit = 0;
                for f = 1:nFold
                    sim_id = 1000*ex+f;
                    oFileName = [talabasepath num2str(sim_id) filesep filelist{k} '.beats.txt'];
                    if exist(oFileName, 'file')
                        oFiles{k} = oFileName;
                        gotit = 1;
                        break;
                    end
                end
%                 annout = dlmread(oFiles{k});
%                 annout(:,2) = round(10*(annout(:,2) - floor(annout(:,2))));
                if gotit
                    fp1 = fopen(oFiles{k},'rt');
                    anntemp = textscan(fp1,'%s %s\n');
                    fclose(fp1);
                    annout(:,1) = str2double(anntemp{1});
                    annout(:,2) = str2double(anntemp{2});
                    opSamas = annout((annout(:,2) == 1),1);
                    opBeats = annout(:,1);
                    % Read tempo
                    % res(k).bpm = dlmread(fullfile(filepath, [fname{k} '.bpm']));
                    % Read output meter
                    fpm = fopen([talabasepath num2str(sim_id) filesep filelist{k} '.meter.txt'],'rt');
                    temp = textscan(fpm,'%s');
                    res(k).meter = temp{1}{1};
                    fclose(fpm);
                    % Read output rhythm/s
                    fpr = fopen([talabasepath num2str(sim_id) filesep filelist{k} '.rhythm.txt'],'rt');
                    temp = textscan(fpr,'%s');
                    res(k).rhythm = [temp{1}{:}];
                    fclose(fpr);
                    % Tempo Estimated
                    temp2 = sort(diff(opBeats));
                    lenn2 = length(opBeats);
                    res(k).bpm = 60./median(temp2(round(lenn2/10):round(0.9*lenn2)));
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
                    fprintf('Exp-%d: Processing file... %s\n',ex, oFiles{k});
                    clear annout ann
                else
                    res(k).bpm = nan;
                    res(k).meter = nan;
                    res(k).rhythm = nan;
                    % Sama metrics
                    res(k).sama.pScore = nan;
                    res(k).sama.fMeas = nan; 
                    res(k).sama.precision = nan; 
                    res(k).sama.recall = nan; 
                    res(k).sama.Ameas = nan; 
                    res(k).sama.infoGain = nan;
                    res(k).sama.cmlC = nan;
                    res(k).sama.cmlT = nan;
                    res(k).sama.amlC = nan;
                    res(k).sama.amlT = nan;
                    % Beat metrics
                    res(k).beat.pScore = nan;
                    res(k).beat.fMeas = nan;
                    res(k).beat.precision = nan;
                    res(k).beat.recall = nan;
                    res(k).beat.Ameas = nan;
                    res(k).beat.infoGain = nan;
                    res(k).beat.cmlC = nan; 
                    res(k).beat.cmlT = nan; 
                    res(k).beat.amlC = nan; 
                    res(k).beat.amlT = nan;
                    fprintf('Exp-%d: Did not find file while processing file... %s\n',ex, filelist{k});
                end                
            end
            % First sort the results on fileID
            [fileID sortInd] = sort(fileID,'ascend');
            res = res(sortInd);
            talaIndicator = talaIndicator(sortInd);
            fileTalaName = fileTalaName(sortInd);
            fileNum = fileNum(sortInd);
            fname = filelist(sortInd);
            % Overall results for each tala
            sm = [res.sama]; s.pScore = [sm.pScore]; s.fMeas = [sm.fMeas];
            s.precision = [sm.precision]; s.recall = [sm.recall]; s.infoGain = [sm.infoGain];
            s.cmlC = [sm.cmlC]; s.cmlT = [sm.cmlT]; s.amlC = [sm.amlC]; s.amlT = [sm.amlT];
            bt = [res.beat]; b.pScore = [bt.pScore]; b.fMeas = [bt.fMeas];    
            b.precision = [bt.precision]; b.recall = [bt.recall]; b.infoGain = [bt.infoGain];
            b.cmlC = [bt.cmlC]; b.cmlT = [bt.cmlT]; b.amlC = [bt.amlC]; b.amlT = [bt.amlT];
            % Write overall results to a file
            colHead1 = 'File,Tala,MedianTempo,EstMeter,EstRhythm,EstTempo,sfMeas,sPrec,sRecall,';
            colHead2 = 'sCMLt,sAMLt,sInfoGain,bfMeas,bPrec,bRecall,bCMLt,bAMLt,bInfoGain';
            fp = fopen([talabasepath exptName '_' talaName{t} '_nPatts_' num2str(numPatts(p)) '_' num2str(ex) '_allResults.txt'], 'wt');
            fprintf(fp, '%s\n', [colHead1 colHead2]);
            for k = 1:length(filelist)
                fprintf(fp, '%s,%s,%.2f,%s,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n',...
                    fname{k},fileTalaName{k},res(k).bpmGT,res(k).meter,res(k).rhythm,...
                    res(k).bpm,s.fMeas(k),s.precision(k),s.recall(k),s.cmlT(k),s.amlT(k),s.infoGain(k),...
                    b.fMeas(k),b.precision(k),b.recall(k),b.cmlT(k),b.amlT(k),b.infoGain(k));
            end
            fclose(fp);
            sMat = [];
            bMat = [];
            samaResults(t).patt(p).expt(ex).allres = res;
            samaResults(t).patt(p).expt(ex).op = s;
            beatResults(t).patt(p).expt(ex).op = b;
            clear res s b oFiles fname fileTalaName fileID talaIndicator fileNum
        end
    end
end
%% Now to generate the mean column for the excel sheet
opFullCol = [];
for m = 1:2
    if m == 2
        samaResults = beatResults;
    end
    for p = 1:length(numPatts)
        fMeasAccFull = [];
        precAccFull = [];
        recallAccFull = [];
        cmltAccFull = [];
        amltAccFull = [];
        infoGainAccFull = [];
        for t = 1:length(talaName)
            for ex = 1:nExp
                % f-Meas
                fMeasVal = samaResults(t).patt(p).expt(ex).op.fMeas';
                fMeas(t+1,ex) = nanmean(fMeasVal);
                fMeasAcc(:,ex) = fMeasVal;
                % 
                precVal = samaResults(t).patt(p).expt(ex).op.precision';
                prec(t+1,ex) = nanmean(precVal);
                precAcc(:,ex) = precVal;
                % 
                recallVal = samaResults(t).patt(p).expt(ex).op.recall';
                recall(t+1,ex) = nanmean(recallVal);
                recallAcc(:,ex) = recallVal;
                % 
                cmltVal = samaResults(t).patt(p).expt(ex).op.cmlT';
                cmlt(t+1,ex) = nanmean(cmltVal);
                cmltAcc(:,ex) = cmltVal;
                %
                amltVal = samaResults(t).patt(p).expt(ex).op.amlT';
                amlt(t+1,ex) = nanmean(amltVal);
                amltAcc(:,ex) = amltVal;
                %
                infoGainVal = samaResults(t).patt(p).expt(ex).op.infoGain';
                infoGain(t+1,ex) = nanmean(infoGainVal);
                infoGainAcc(:,ex) = infoGainVal;
            end
            fMeasAccFull = [fMeasAccFull; fMeasAcc];
            precAccFull = [precAccFull; precAcc];
            recallAccFull = [recallAccFull; recallAcc];
            cmltAccFull = [cmltAccFull; cmltAcc];
            amltAccFull = [amltAccFull; amltAcc];
            infoGainAccFull = [infoGainAccFull; infoGainAcc];
            clear fMeasAcc precAcc recallAcc cmltAcc amltAcc infoGainAcc
        end
        fMeas(1,:) = nanmean(fMeasAccFull);
        prec(1,:) = nanmean(precAccFull);
        recall(1,:) = nanmean(recallAccFull);
        cmlt(1,:) = nanmean(cmltAccFull);
        amlt(1,:) = nanmean(amltAccFull);
        infoGain(1,:) = nanmean(infoGainAccFull);
        zz = zeros(1,nExp);
        opCol(:,p) = nanmean([fMeas; zz; prec; zz; recall; zz; cmlt; zz; amlt; zz; infoGain; zz; zz],2);
    end
    opFullCol = [opFullCol; opCol];
    clear opCol;
end

for p = 1:length(numPatts)
    dlmwrite([bpath exptName filesep 'opColumn_nPatts_' num2str(numPatts(p)) '.txt'], opFullCol(:,p), 'precision', '%.2f');
end

            
%             sMat = [sMat; [mean(s.fMeas) mean(s.precision) mean(s.recall) mean(s.cmlT) ...
%                 mean(s.amlT) mean(s.infoGain)]];
%             bMat = [bMat; [mean(b.fMeas) mean(b.precision) mean(b.recall) mean(b.cmlT)...
%                 mean(b.amlT) mean(b.infoGain)]];
% 
%         for k = unique(talaIndicator)%1:length(unique(talaIndicator))
%             chosen = find(talaIndicator == k);
%             fprintf('Taala: %s\n',talaName{k});
%             sMat = [sMat; [mean(s.fMeas(chosen)) mean(s.precision(chosen)) ...
%                 mean(s.recall(chosen)) mean(s.cmlT(chosen)) ...
%                 mean(s.amlT(chosen)) mean(s.infoGain(chosen))]];
%             bMat = [bMat; [mean(b.fMeas(chosen)) mean(b.precision(chosen)) ...
%                 mean(b.recall(chosen)) mean(b.cmlT(chosen))...
%                 mean(b.amlT(chosen)) mean(b.infoGain(chosen))]];
%         end
%         sMat = round(sMat*100)/100;
%         bMat = round(bMat*100)/100;
%         dlmwrite([exppath exptName '_' num2str(ex) '_opResults.txt'], sMat, '-append', 'precision','%.2f');
%         dlmwrite([exppath exptName '_' num2str(ex) '_opResults.txt'], bMat, '-append', 'precision','%.2f');
%         % Also generate the "column" for excel
%         sMat = [sMat; zeros(1,size(sMat,2))];
%         bMat = [bMat; zeros(1,size(bMat,2))];
%         dlmwrite([exppath exptName '_' num2str(ex) '_opCol.txt'], sMat(:), '-append', 'precision','%.2f');
%         dlmwrite([exppath exptName '_' num2str(ex) '_opCol.txt'], 0, '-append', 'precision','%.2f');
%         dlmwrite([exppath exptName '_' num2str(ex) '_opCol.txt'], bMat(:), '-append', 'precision','%.2f');
%         clear res sMat bMat
%     end
