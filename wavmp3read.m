function [x Fs] = wavmp3read(fpath)
[~, ~, ext] = fileparts(fpath);
if ext == '.wav'
    [x, fs] = wavread(fpath);
elseif ext == '.mp3'
    
    [x, fs] = wavread(fpath);
    
else
    error('Unsupported format');
end
