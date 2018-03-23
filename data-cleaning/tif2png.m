function [  ] = tif2png( nodules_regex )
%TIF2PNG  Transforms all nodule images that match given regex into PNG images.

if nargin < 1
    nodules_regex = '/media/maciej/Thyroid/thyroid-nodules/Nodules/*.TIF';
end

nodules_us = dir(nodules_regex);

for i = 1:numel(nodules_us)
    
    img_path = fullfile(nodules_us(i).folder, nodules_us(i).name);
    img = rgbread(img_path);
    
    imwrite(img, [img_path(1:end-3), 'PNG']);
    
    delete(img_path);
    
end
