function [outmat,outmatDB] =  ISMIR2016_eval(dataset,remote,dbeval)
% this is the evaluation script that compares the meter tracking outputs of
% our approach from ISMIR 2016, and compares them with the reference
% annotations. Code was initially taken from Ajay Srinivasamurthy. It was
% extended to work with the local network directories at OFAI.
%
% by Andre Holzapfel, 2016
piecewise = 1;
bayes_pathinit;
annpath = [dpath 'annotations/beats/'];
exptName = 'PeakPickAct_8_2_2016_rev451';
exppath = [bpath filesep];%[bpath exptName filesep];
% talaID = [22];
% talaID = [11:18];
nExp = 1;
nFold = 1;
numPatts = [1];
% talaName = {'jhap'};
% talaName = {'cretan'};
% talaName = {'ChaChaCha', 'Jive' , 'Quickstep', 'Rumba' , 'Samba' , 'Tango', 'VienneseWaltz', 'Waltz'};
allTaalaMat = [];
outmat = zeros(length(talaName)+1,4);
allTaalaMatDB = [];
outmatDB = zeros(length(talaName)+1,3);
for t = 1:length(talaName)
    fpp = fopen([dpath 'filelist_' talaName{t} '.txt']);
    filelist = textscan(fpp,'%s');
    filelist = filelist{1};
    fclose(fpp);
    for p = 1:length(numPatts)
        talabasepath = [exppath];% talaName{t} filesep 'nPatts_' num2str(numPatts(p)) filesep];
        for ex = 1:nExp
            for k = 1:length(filelist)
                % Read the ground truth first
                EvaluatedFileCounter = 0;
                if strcmp(dataset,'HMDf')
                    annFile = [filelist{k} '.beats'];
                    talaIndicator(k) = find(ismember(talaID, str2num(annFile(4:5))));
                    fileID(k) = str2num(annFile(4:8));
                    fileNum(k) = str2num(annFile(1:2));
                elseif strcmp(dataset,'Ballroom')
                    annFile = [filelist{k}(1:end-4) '.beats'];
                    talaIndicator(k) = find(ismember(talaID, str2num(annFile(1:2))));
                    fileID(k) = str2num(annFile(4:9));%(annFile(4:8));
                    fileNum(k) = str2num(annFile(7:9));%(annFile(1:2));
                else
                    annFile = [filelist{k}(1:end-4) '.beats'];
                    talaIndicator(k) = find(ismember(talaID, str2num(annFile(1:2))));
                    fileID(k) = str2num(annFile(1:5));%(annFile(4:8));
                    fileNum(k) = str2num(annFile(4:5));%(annFile(1:2));
                end
                fileTalaName{k} = talaName{talaIndicator(k)};
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
                    %sim_id = 1000*ex+f;
                    if strcmp(dataset,'HMDf')
                        if remote == 1
                            oFileName = [talabasepath 'beats/' filelist{k} '.txt'];
                            oFileNameDB = [talabasepath 'downbeats/' filelist{k} '.txt'];
                        else
                            oFileName = [talabasepath filelist{k} '.beats.txt'];
                        end
                    else
                        if remote == 1
                            if strcmp(dataset,'Ballroom')
                                oFileName = [talabasepath 'beats/' filelist{k}(11:end-4) '.txt'];
                                oFileNameDB = [talabasepath 'downbeats/' filelist{k}(11:end-4) '.txt'];
                            else
                                oFileName = [talabasepath 'beats/' filelist{k}(1:end-4) '.txt'];
                                oFileNameDB = [talabasepath 'downbeats/' filelist{k}(1:end-4) '.txt'];
                            end
                        else
                            oFileName = [talabasepath filelist{k}(1:end-4) '.beats.txt'];%[talabasepath num2str(sim_id) filesep filelist{k} '.beats.txt'];
                        end
                    end
                    if exist(oFileName, 'file')
                        oFiles{k} = oFileName;
                        if remote == 1
                            oFilesDB{k} = oFileNameDB;
                        end
                        gotit = 1;
                        break;
                    end
                end
%                 annout = dlmread(oFiles{k});
%                 annout(:,2) = round(10*(annout(:,2) - floor(annout(:,2))));
                if gotit
                    EvaluatedFileCounter = EvaluatedFileCounter +1;
                    fp1 = fopen(oFiles{k},'rt');
                    if remote == 1
                        anntemp = textscan(fp1,'%f %f %s\n');
                        annout(:,1) = unique([anntemp{1};anntemp{2}]);
                        fp2 = fopen(oFilesDB{k},'rt');
                        anntemp = textscan(fp2,'%f %f %s\n');
                        fclose(fp2);
                    else
                        anntemp = textscan(fp1,'%s %s\n');
                        annout(:,1) = str2double(anntemp{1});
                    end
                    fclose(fp1);
                    opBeats = annout(:,1);
                    if dbeval
                        if remote == 1
                            opSamas = unique([anntemp{1};anntemp{2}]);
                        else
                            annout(:,2) = str2double(anntemp{2});
                            opSamas = annout((annout(:,2) == 1),1);
                        end
                        %opSamas = annout(:,1);
                        [res(k).sama.fMeas res(k).sama.precision res(k).sama.recall ] = be_fMeasure(samas,opSamas);
                    end
                    % Tempo Estimated
                    temp2 = sort(diff(opBeats));
                    lenn2 = length(opBeats);
                    res(k).bpm = 60./median(temp2(round(lenn2/10):round(0.9*lenn2)));
                    % Beat metrics
                    res(k).beat.pScore = be_pScore(beats,opBeats);
                    [res(k).beat.fMeas res(k).beat.precision res(k).beat.recall res(k).beat.Ameas]...
                        = be_fMeasure(beats,opBeats);
                    res(k).beat.infoGain = be_informationGain(beats,opBeats);
                    [res(k).beat.cmlC res(k).beat.cmlT res(k).beat.amlC res(k).beat.amlT]...
                        = be_continuityBased(beats,opBeats);
                    %fprintf('Exp-%d: Processing file... %s\n',ex, oFiles{k});
                    clear annout ann
                else
                    res(k).bpm = nan;
                    if dbeval
                        res(k).sama.fMeas = nan;
                        res(k).sama.precision = nan; 
                        res(k).sama.recall = nan; 
                    end
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
                    %fprintf('Exp-%d: Did not find file while processing file... %s\n',ex, filelist{k});
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
            bt = [res.beat]; b.pScore = [bt.pScore]; b.fMeas = [bt.fMeas];    
            b.precision = [bt.precision]; b.recall = [bt.recall]; b.infoGain = [bt.infoGain];
            b.cmlC = [bt.cmlC]; b.cmlT = [bt.cmlT]; b.amlC = [bt.amlC]; b.amlT = [bt.amlT];
            % Write overall results to a file
            %colHead1 = 'File,Tala,MedianTempo,EstMeter,EstRhythm,EstTempo,sfMeas,sPrec,sRecall,';
            colHead1 = 'File,Tala,MedianTempo,EstTempo,';
            colHead2 = 'sFm,sPrec,sRec,bfMeas,bCMLt,bAMLt,bInfoGain';
            fp = fopen([talabasepath exptName '_' talaName{t} '_allResults.txt'], 'wt');
            fprintf(fp, '%s\n', [colHead1 colHead2]);
            thisTaalaMat = zeros(EvaluatedFileCounter,4);
            thisTaalaMatDB = zeros(EvaluatedFileCounter,3);
            index = 1;
            for k = 1:length(filelist)
                if ~isnan(res(k).beat.fMeas)
                    %fprintf(fp, '%s,%s,%.2f,%s,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n',...
                    %    fname{k},fileTalaName{k},res(k).bpmGT,res(k).meter,res(k).rhythm,...
                    %    res(k).bpm,s.fMeas(k),s.precision(k),s.recall(k),s.cmlT(k),s.amlT(k),s.infoGain(k),...
                    %    b.fMeas(k),b.precision(k),b.recall(k),b.cmlT(k),b.amlT(k),b.infoGain(k));
                    fprintf(fp, '%s,%s,%.2f(est:%.2f),%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n',...
                        fname{k},fileTalaName{k},res(k).bpmGT,res(k).bpm,...
                        res(k).sama.fMeas,res(k).sama.precision,res(k).sama.recall,...
                        b.fMeas(k),b.cmlT(k),b.amlT(k),b.infoGain(k));
                    thisTaalaMat(index,:) = [b.fMeas(k),b.cmlT(k),b.amlT(k),b.infoGain(k)];
                    thisTaalaMatDB(index,:) = [res(k).sama.fMeas,res(k).sama.precision,res(k).sama.recall];
                    index = index +1;
                end
            end
            fclose(fp);
        end
        samaResults(t).patt(p).expt(ex).allres = res;
        beatResults(t).patt(p).expt(ex).op = b;
        clear res s b oFiles fname fileTalaName fileID talaIndicator fileNum
    end
    allTaalaMat = [allTaalaMat ; thisTaalaMat];
    outmat(t,:) = mean(thisTaalaMat);
    allTaalaMatDB = [allTaalaMatDB ; thisTaalaMatDB];
    outmatDB(t,:) = mean(thisTaalaMatDB);
end
outmat(t+1,:)=mean(allTaalaMat)
outmatDB(t+1,:)=mean(allTaalaMatDB)
csvwrite([talabasepath exptName '_meanResults.txt'], [outmatDB outmat]);
csvwrite([talabasepath exptName '_AllTaalaResults.txt'], [allTaalaMatDB allTaalaMat]);