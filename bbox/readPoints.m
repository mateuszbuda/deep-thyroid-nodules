function pts = readPoints(image, n)
%readPoints   Read manually-defined points from image
%   POINTS = READPOINTS(IMAGE) displays the image in the current figure,
%   then records the position of each click of button 1 of the mouse in the
%   figure, and stops when another button is clicked. The track of points
%   is drawn as it goes along. The result is a 2 x NPOINTS matrix; each
%   column is [X; Y] for one point.
% 
%   POINTS = READPOINTS(IMAGE, N) reads up to N points only.

if nargin < 2
    n = Inf;
    pts = zeros(2, 0);
else
    pts = zeros(2, n);
end

% display image
imshow(image);

% and keep it there while we plot
hold on;

k = 0;

while 1
    % get a point
    [xi, yi, but] = myginput(1, 'crosshair');
    % stop if not button 1
    if ~isequal(but, 1)
        break
    end
    k = k + 1;
    pts(1,k) = xi;
    pts(2,k) = yi;
    
    plot(xi, yi, 'rs', 'markers', 33);
    
    if isequal(k, n)
        break
    end
end

hold off;

if k < size(pts,2)
    pts = pts(:, 1:k);
end
