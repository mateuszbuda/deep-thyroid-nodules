function [ rgb ] = rgbread( img_path )
%RGBREAD Reads image from given path and transforms it to RGB image if
%needed

[img, map] = imread(img_path);

if map
    rgb = ind2rgb(img, map);
else
    if size(img, 3) == 1
        rgb = cat(3, img, img, img);
    else
        rgb = img;
    end
end
