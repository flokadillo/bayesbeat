function [ data, error ] = load_annotations_bt( filename, ann_type )
% [data, error] = load_annotations_bt( filename, dooutput )
%
%   Loads annotations according to the extension of the file from file.
%   Supports .beats, .bpm, .meter, .rhythm, .dancestyle, .onsets, .section
%   The following cases are supported
%   1) The extension does not match the supported ones. Please specify
%   ann_type in this case
%   2) The path represents the audio-folder (dataset/audio). Here, the
%   path is updated to point to dataset/annotations/xxx
% ----------------------------------------------------------------------
% INPUT Parameter:
% filename          : filename of annotation file including extension
% datatype          : [optional, only if extension is not standard]
%                       e.g., 'beats', 'bpm', 'onsets', 'meter', ...
%
% OUTPUT Parameter:
% data              : annotations
% error             : > 0 if some of the requested entities could not be
%                       loaded
%
% 16.05.2014 by Florian Krebs
% ----------------------------------------------------------------------

% supported extensions:
sup_ext = {'.beats', '.meter', '.onsets', '.bpm', '.dancestyle', '.rhythm', '.section'};
% get extension of filename
[path, fname, ext] = fileparts(filename);
if sum(strcmp(sup_ext, ext)) > 0
    % ext is standard, set ann_type = ext
    ann_type = ext(2:end);
end
path = strrep(path, 'audio', ['annotations/', ann_type]);
filename = fullfile(path, [fname, '.', ann_type]);
% check if file exists
if ~exist(filename, 'file')
    fprintf('    WARNING: load_annotations_bt: %s not found\n', filename);
    data = [];
    error = 1;
    return;
end

error = 0;
if strcmp(ext, '.meter') || strcmp(ann_type, 'meter')
    % Load meter
    fid = fopen(filename, 'r');
    data = double(cell2mat(textscan(fid, '%d%d', 'delimiter', '/')));
    fclose(fid);
    if data(2) == 0, data = data(1); end
elseif strcmp(ext, '.section') || strcmp(ann_type, 'section')
    % Load section
    fid = fopen(filename, 'r');
    data = double(cell2mat(textscan(fid, '%d', 'delimiter', ',')));
elseif strcmp(ext, '.onsets') || strcmp(ann_type, 'onsets')
    % Load onset annotations
    data = load('-ascii', filename);
elseif strcmp(ext, '.beats') || strcmp(ann_type, 'beats')
    % Load beat annotations
    fid = fopen(filename, 'r');
    temp = textscan(fid, '%f64%s', 'delimiter', '\t');
    data = temp{1};
    if size(temp, 2) == 2
        % get bar id and beat number
        temp2 = cellfun(@(x) textscan(x, '%s%s', 'Delimiter', '.'), temp{2}, ...
            'UniformOutput', 0);
        if strfind(temp{2}{1}, '.') > 0 % bar_id present in annotation file
            bar_id = cellfun(@(x) str2num(x), cellfun(@(x) x{1}, temp2));
            beat_number = cellfun(@(x) str2num(x), cellfun(@(x) x{2}, temp2));
        else % no bar_id given, set bar_id to nan
            beat_number = cellfun(@(x) str2num(x), cellfun(@(x) x{1}, temp2));
            bar_id = nan(size(beat_number));
        end
%         data = [data, bar_id, beat_number];
        data = [data, beat_number];
    end
    fclose(fid);
elseif strcmp(ext, '.bpm') || strcmp(ann_type, 'bpm')
    % Load Tempofile
    data = load('-ascii', filename);
elseif strcmp(ext, '.dancestyle') || strcmp(ann_type, 'dancestyle')
    fid = fopen(filename, 'r');
    data = textscan(fid, '%s');
    data = data{1}{1};
    fclose(fid);
elseif strcmp(ext, '.rhythm') || strcmp(ann_type, 'rhythm')
    fid = fopen(filename, 'r');
    data = textscan(fid, '%s');
    data = data{1}{1};
    fclose(fid);
end
end

