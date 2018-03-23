function [ ] = bbox(images_regex, bboxes_out)

if nargin < 2
    bboxes_out = '/media/maciej/Thyroid/thyroid-nodules/detection/Calipers/';
end
if nargin < 1
    images_regex = '/media/maciej/Thyroid/thyroid-nodules/Nodules/*.PNG';
end

nodules_us = dir(images_regex);

for i = 1:numel(nodules_us)
    
    in_path = fullfile(nodules_us(i).folder, nodules_us(i).name);
    out_name = [nodules_us(i).name(1:end-3), 'csv'];
    out_path = fullfile(bboxes_out, out_name);
    
    if exist(out_path, 'file') == 2
        continue;
    end
    
    % read
    disp(in_path);
    pts = readPoints(in_path);
    
    % transform to (row, column) coordinates format
    pts = round(flip(pts)');
    
    % write
    csvwrite(out_path, pts);
    
end

close
