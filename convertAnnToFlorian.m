DIR = '/home/hannover/Documents/databases/ethno_rhythm_ismir';
D = dir(DIR);k=0;
clear styleDir;
for i=1:length(D)
    if(D(i).isdir & ~(D(i).name(1) == '.'))
        D2 = dir(fullfile(DIR,D(i).name));
        for j=1:length(D2)
            if(D2(j).isdir & ~(D2(j).name(1) == '.'))
                path = fullfile(DIR,D(i).name,D2(j).name);
                mkdir(fullfile(path,'annotations'));
                mkdir(fullfile(path,'annotations','beats'));
                mkdir(fullfile(path,'annotations','bpm'));
                mkdir(fullfile(path,'annotations','meter'));
                TL = dir([fullfile(path,'Annotations') '/*downb_fin.txt']);
                for jj = 1:length(TL)
                    fid1 = fopen(fullfile(path,'Annotations',TL(jj).name),'r');
                    [tmp] = textscan(fid1,'%f');
                    fclose(fid1);
                    dbs = tmp{1};
                    fid1 = fopen(fullfile(path,'Annotations',[TL(jj).name(1:end-13) 'regu1_fin.txt']),'r');
                    [tmp] = textscan(fid1,'%f');
                    fclose(fid1);
                    beats = tmp{1};
                    [ans1,ans2]=ismember(dbs,beats);
                    meter = unique(diff(ans2));%compensate for mistakes
                    if length(meter)>1
                        error(sprintf('Problematic annotations in %s: file named: %s, number %i',path,TL(jj).name(1:end-18),jj))
                        %fid = fopen(fullfile(path,'Annotations',[TL(jj).name(1:end-13) 'regu1_fin.txt']),'w');
                        %for m=1:length(beats)
                        %    fprintf(fid, '%.3f\n',beats(m));
                        %end
                        %fclose(fid);
                    end
                    offset = ans2(1);
                    fid = fopen(fullfile(path,'annotations/beats',[TL(jj).name(1:end-18) '.beats']),'w');
                    for m=1:length(beats)
                        fprintf(fid, '%.3f\t %i.%i\n',beats(m),ceil((m+offset-1)/meter),mod(meter-offset+m,meter)+1);
                    end
                    fclose(fid);
                    tempo_bpm = 60/median(diff(beats));
                    fid = fopen(fullfile(path,'annotations/bpm',[TL(jj).name(1:end-18) '.bpm']),'w');
                    fprintf(fid, '%.3f',tempo_bpm);
                    fclose(fid);
                    if meter>4
                        denum = 8;
                    else
                        denum = 4;
                    end
                    fid = fopen(fullfile(path,'annotations/meter',[TL(jj).name(1:end-18) '.meter']),'w');
                    fprintf(fid, '%i/%i',meter,denum);
                    fclose(fid);
                end
            end
        end
    end
end
