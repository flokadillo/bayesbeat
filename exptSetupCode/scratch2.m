clear
close all
clc
pp = [1 2 4];
talas = {'adi' 'rupaka' 'mChapu' 'kChapu'};
oValsLarge = [];
for t = 1:length(talas)
    cd(['HMM_viterbi_HMM_' talas{t}]);
    for p = 1:length(pp)
        cd(['nPatts_' num2str(pp(p))]);
        files = dir('*opResults.txt');
        for k = 1:length(files)
            vals = dlmread(files(k).name);
            sVals(k,:) = vals(1,:);
            bVals(k,:) = vals(3,:);
        end
        oVals(:,p) = [mean(sVals) 0 mean(bVals)]';
        cd ..
    end
    oValsLarge = [oValsLarge; oVals; zeros(1,size(oVals,2))];
    clear oVals
    cd ..
end
dlmwrite('scr.txt',oValsLarge,'precision','%.2f');