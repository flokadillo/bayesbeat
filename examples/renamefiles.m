% messy collection of scripts that were used to deal with the inconsistent
% naming schemes applied by the authors of the ISMIR 2016 paper...
% by Andre Holzapfel
for i = 1:length(file_list)
    for j = 1:length(actlist)
        [~,wavname,~]=fileparts(file_list{i});
        [~,featname,endung]=fileparts(actlist(j).name);
        if ~isempty(findstr(featname,wavname))
            command = ['mv ./' actlist(j).name ' ./' wavname  endung];
            system(command);
        end
    end
end

% for i = 1:length(file_list)
%     [~,fname,~]=fileparts(file_list{i});
%     i
%     annpath = [audio_path '/annotations/meter/'];
%     annFile = [fname '.meter'];
%     fid = fopen([annpath annFile],'r');
%     data = double(cell2mat(textscan(fid, '%d%d', 'delimiter', '/')));
%     fclose(fid);
%     if 1%data(1)==12
%         data(2) = 4;
%         fid = fopen([annpath annFile],'w');
%         fprintf(fid,'%i/%i',data(1),data(2));
%         fclose(fid);
%     end
% end

%in the ballroom dataset files were renamed. Rewrite the file_list
%accordingly
% in_file = '/home/hannover/Documents/databases/Ballroom/val.lab';
% fid = fopen(in_file, 'r');
% file_list = textscan(fid, '%s', 'delimiter', '\n');
% file_list = file_list{1};
% fclose(fid);
% path = '/home/hannover/Documents/databases/Ballroom/audio/';
% actlist = dir([path '*.wav']);
% fid = fopen([in_file '.tmp'], 'w');
% for i = 1:length(file_list)
%     for j = 1:length(actlist)
%         [~,filename,~] = fileparts(file_list{i});
%         if strfind(actlist(j).name,filename)
%             fprintf(fid,'%s\n',fullfile(path,actlist(j).name));
%         end
%     end
% end

% in_file = '/home/hannover/Documents/databases/CMCMDa_v2/CarnaticsFromThomasTrainList.lab';
% fid = fopen(in_file, 'r');
% file_list = textscan(fid, '%s', 'delimiter', '\n');
% file_list = file_list{1};
% path = '/home/hannover/Documents/databases/CMCMDa_v2/audio/';
% fid2 = fopen([in_file '.local'], 'w');
% for i = 1:length(file_list)
%     file_list{i} = [path file_list{i}];
%     fprintf(fid2,'%s\n',file_list{i});
% end
% fclose(fid2)
% 
% in_file = '/home/hannover/Documents/databases/RemoteDragonetti/Documents/databases/Ballroom/train_wpath.lab';
% fid = fopen(in_file, 'r');
% file_list = textscan(fid, '%s', 'delimiter', '\n');
% file_list = file_list{1};
% fclose(fid)
% fid1 = fopen([in_file '.1'], 'w');
% fid2 = fopen([in_file '.2'], 'w');
% fid3 = fopen([in_file '.3'], 'w');
% fid4 = fopen([in_file '.4'], 'w');
% fid5 = fopen([in_file '.5'], 'w');
% fid6 = fopen([in_file '.6'], 'w');
% fid7 = fopen([in_file '.7'], 'w');
% fid8 = fopen([in_file '.8'], 'w');
% for i = 1:length(file_list)
%     if ~isempty(findstr('/11_',file_list{i}))
%         fprintf(fid1,'%s\n',file_list{i});
%     elseif ~isempty(findstr('/12_',file_list{i}))
%         fprintf(fid2,'%s\n',file_list{i});
%     elseif ~isempty(findstr('/13_',file_list{i}))
%         fprintf(fid3,'%s\n',file_list{i});
%     elseif ~isempty(findstr('/14_',file_list{i}))
%         fprintf(fid4,'%s\n',file_list{i});
%     elseif ~isempty(findstr('/15_',file_list{i}))
%         fprintf(fid5,'%s\n',file_list{i});
%     elseif ~isempty(findstr('/16_',file_list{i}))
%         fprintf(fid6,'%s\n',file_list{i});
%     elseif ~isempty(findstr('/17_',file_list{i}))
%         fprintf(fid7,'%s\n',file_list{i});
%     elseif ~isempty(findstr('/18_',file_list{i}))
%         fprintf(fid8,'%s\n',file_list{i});
%     else
%         error('String not found')
%     end
% end
% fclose(fid1)
% fclose(fid2)
% fclose(fid3)
% fclose(fid4)
% fclose(fid5)
% fclose(fid6)
% fclose(fid7)
% fclose(fid8)

%check file lists for distinctness!
% dataset = 'Ballroom';
% taala = 6;
% piecewise = 0;
% remote = 0;%check
% focusTempoStates = 0;
% bayes_pathinit;
% fid = fopen(in_file, 'r');
% file_list = textscan(fid, '%s', 'delimiter', '\n');
% file_list = file_list{1};
% fclose(fid);
% fid = fopen(train_lab, 'r');
% train_list = textscan(fid, '%s', 'delimiter', '\n');
% train_list = train_list{1};
% fclose(fid);
% for i = 1:length(file_list)
%     i
% for j = 1:length(train_list)
% if strcmp(file_list{i},train_list{j})
% sprintf('Problem: %i %i\n',i,j)
% end
% end
% end