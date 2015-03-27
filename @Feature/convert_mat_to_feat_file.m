function [] = convert_mat_to_feat_file(fpath, output_ext, input_fr)
if nargin < 3
   input_fr = 100; 
end
if isempty(strfind(output_ext, '.'))
    output_ext = ['.', output_ext];
end
listing = dir(fullfile(fpath, '*.mat'));
for i=1:length(listing)
    load(fullfile(fpath, listing(i).name));
    save_fln = fullfile(fpath, strrep(listing(i).name, '.mat', output_ext));
    fid=fopen(save_fln,'w');
    fprintf(fid,'FRAMERATE: %.4f\nDIMENSION: %i\n', input_fr, length(data));
    fprintf(fid,'%d ',data');
    fclose(fid);
    fprintf('    Feature saved to %s\n', save_fln);
end

end