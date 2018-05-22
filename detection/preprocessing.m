function [ ] = preprocessing( images_regex )
%PREPROCESSING Preprocessing

if nargin < 1
    images_regex = '/media/maciej/Thyroid/thyroid-nodules/detection/Nodules/*.PNG';
end

image_dir = dir(images_regex);

for i = 1:numel(image_dir)
    
    im_path = fullfile(image_dir(i).folder, image_dir(i).name);
    image = imread(im_path);
    
    image = remove_artifacts(image);
    
    image = adjust_contrast(image, [0.0, 1.0]);

    imwrite(image, im_path);
    
end

end

function [ image ] = adjust_contrast( image, lims )
%ADJUST_CONTRAST Contrast stretching in range `[0.01, 0.99]`

if nargin < 2
    lims = [0.01, 0.99];
end
image = imadjust(image, stretchlim(image, lims), []);

end

function [ image_out ] = remove_artifacts( image )
%REMOVE_ARTIFACTS Removes large connected component of the same pixel value

if numel(size(image)) > 2
    image = rgb2gray(image); 
end

image_out = image;
image = adjust_contrast(image);

[H, W] = size(image);

intensity_th = 229;
[counts, bins] = imhist(image, 256);

for dummy = 1:3
    [~, max_i] = max(counts((intensity_th + 1):end));
    v = bins(intensity_th + max_i);
    bw = image == v;
    bw = imdilate(bw, [0 1 0; 1 1 1; 0 1 0]);
    cc = bwconncomp(bw);
    stats = regionprops(cc, 'MajorAxisLength', 'MinorAxisLength');

    for i = 1:cc.NumObjects
        [y, x] = ind2sub(cc.ImageSize, cc.PixelIdxList{i});
        % cc must take at least 15% of image height or width
        % (handles small region of high intensity, e.g. calcifications)
        if max(y) - min(y) > 0.15 * H || max(x) - min(x) > 0.15 * W
            valid_cc = true;
            % cc bbox should be either very large (take > 25% of area) or
            % have proportions of dimensions > 15%
            % (large regions handle connected vertical and horizontal axes)
            cc_w = max(x) - min(x);
            cc_h = max(y) - min(y);
            if cc_w * cc_h < 0.25 * W * H
                % skip cc if proportions of dimensions > 15%
                if stats(i).MinorAxisLength / stats(i).MajorAxisLength > 0.15
                    continue;
                end
                % each point must be within 25% of image size from its sides
                for k = 1:numel(cc.PixelIdxList{i})
                    if (x(k) > 0.25 * W && x(k) < 0.75 * W) && (y(k) > 0.25 * H && y(k) < 0.75 * H)
                        valid_cc = false;
                        break;
                    end
                end
            end
            if valid_cc
                for k = 1:numel(cc.PixelIdxList{i})
                    y_win = max([1, y(k)-2]):min([H, y(k)+2]);
                    x_win = max([1, x(k)-2]):min([W, x(k)+2]);
                    values = image_out(y_win, x_win);
                    values_bw = bw(y_win, x_win);
                    values = values(values_bw == 0);
                    if numel(values) > 0
                        pixel_val = median(values(:));
                        image_out(y(k), x(k)) = pixel_val;
                    else
                        image_out(y(k), x(k)) = 0;
                    end
                end
            end
        end
    end

    counts(intensity_th + max_i) = 0;
end

image_out = cat(3, image_out, image_out, image_out);

end

function [ image ] = remove_empty_slice( image )
%REMOVE_EMPTY_SLICE Removes rows and columns from the bottom and left half 
%of the image that are (almost) black

if numel(size(image)) > 2
    image = rgb2gray(image);
end

[counts, ~] = imhist(image, 256);

ver_proj = sum(image, 2);
ver_th = 0.1 * size(image, 2);
if counts(2) > counts(1)
    ver_th = 1.1 * size(image, 2);
end
ver_proj_mask = ver_proj <= ver_th;
ver_indices = (1:numel(ver_proj))';
ver_indices = ver_indices .* ver_proj_mask;
ver_indices = ver_indices(round(numel(ver_indices) / 2) : end);
ver_indices = ver_indices(ver_indices ~= 0);

image(ver_indices', :) = [];

hor_proj = sum(image, 1);
hor_th = 0.1 * size(image, 1);
if counts(2) > counts(1)
    hor_th = 1.1 * size(image, 1);
end
hor_proj_mask = hor_proj <= hor_th;
hor_indices = 1:numel(hor_proj);
hor_indices = hor_indices .* hor_proj_mask;
hor_indices = hor_indices(round(numel(hor_indices) / 2) : end);
hor_indices = hor_indices(hor_indices ~= 0);

image(:, hor_indices) = [];

image = cat(3, image, image, image);

end
