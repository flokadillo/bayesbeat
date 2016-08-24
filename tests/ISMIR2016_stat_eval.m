function [h,p] = ISMIR2016_stat_eval
% statistical significance evaluation as done for the ISMIR 2016
% by Andre Holzapfel
%Ballroom:peakpick, baseline, cnn-dbn
mat{1,1}=csvread('/home/hannover/Documents/experiments/results/11_2015_DBNonCNN/Ballroom/PeakPickAct_17_3_2016_repeat3/PeakPickAct_8_2_2016_rev451_AllTaalaResults.txt');
mat{1,2}=csvread('/home/hannover/Documents/experiments/results/11_2015_DBNonCNN/Ballroom/HMM_bar_8_2_SFbaseline_rev451/HMM_bar_5_2_2016_rev447_focusTempoState0_AllTaalaResults.txt');
mat{1,3}=csvread('/home/hannover/Documents/experiments/results/11_2015_DBNonCNN/Ballroom/HMM_bar_11_3_focused_7tempiperpeak_repeat3/PeakPickAct_8_2_2016_rev451_AllTaalaResults.txt');
mat{1,4}=csvread('/home/hannover/Documents/experiments/results/11_2015_DBNonCNN/Ballroom/HMM_bar_11_3_tempogiven_repeat3/PeakPickAct_8_2_2016_rev451_AllTaalaResults.txt');
%Hindustani
mat{2,1}=csvread('/home/hannover/Documents/experiments/results/11_2015_DBNonCNN/CMCMDa_v2/PeakPickAct_17_3_2016_repeat2/PeakPickAct_8_2_2016_rev451_AllTaalaResults.txt');
mat{2,2}=csvread('/home/hannover/Documents/experiments/results/11_2015_DBNonCNN/CMCMDa_v2/HMM_bar_8_2_SFbaseline_rev451/HMM_bar_5_2_2016_rev447_focusTempoState0_AllTaalaResults.txt');
mat{2,3}=csvread('/home/hannover/Documents/experiments/results/11_2015_DBNonCNN/CMCMDa_v2/HMM_bar_11_3_focused_7tempiperpeak_repeat3/PeakPickAct_8_2_2016_rev451_AllTaalaResults.txt');
mat{2,4}=csvread('/home/hannover/Documents/experiments/results/11_2015_DBNonCNN/CMCMDa_v2/HMM_bar_9_2_2016_tempogiven/PeakPickAct_8_2_2016_rev451_AllTaalaResults.txt');
%Carnatic
mat{3,1}=csvread('/home/hannover/Documents/experiments/results/11_2015_DBNonCNN/HMDf/PeakPickAct_17_3_2016_repeat2/PeakPickAct_8_2_2016_rev451_AllTaalaResults.txt');
mat{3,2}=csvread('/home/hannover/Documents/experiments/results/11_2015_DBNonCNN/HMDf/HMM_bar_8_2_SFbaseline_rev451/HMM_bar_5_2_2016_rev447_focusTempoState0_AllTaalaResults.txt');
mat{3,3}=csvread('/home/hannover/Documents/experiments/results/11_2015_DBNonCNN/HMDf/HMM_bar_11_3_focused_7tempiperpeak_repeat3/PeakPickAct_8_2_2016_rev451_AllTaalaResults.txt');
mat{3,4}=csvread('/home/hannover/Documents/experiments/results/11_2015_DBNonCNN/HMDf/HMM_bar_9_2_2016_tempogiven/PeakPickAct_8_2_2016_rev451_AllTaalaResults.txt');


%do a simple ttest over all datasets
peakmat = [];basemat = [];thismat = [];thismat_avg = [];
for dataset = 2:3
    peakmat = [peakmat; mat{dataset,1}];
    basemat = [basemat; mat{dataset,2}];
    thismat = [thismat; mat{dataset,3}];
    thismat_avg = [thismat_avg; mat{dataset,4}];
end



[h(1,:),p(1,:)] = ttest2(thismat,basemat);
[h(2,:),p(2,:)] = ttest2(thismat,peakmat);
[h(3,:),p(3,:)] = ttest2(thismat_avg,basemat);
[h(4,:),p(4,:)] = ttest2(thismat_avg,thismat);

[mean(peakmat);mean(basemat);mean(thismat);mean(thismat_avg)]