%-- A demo script for sfm_chanvese.m
%--
%-- The script opens and image, prepares an initialization
%   and shows the final output after segmentation.


root_dir = 'E:\dj\VoxelMorph\PH';
folders = dir(root_dir);
for i = 3:length(folders)
    imgfile = fullfile(root_dir,folders(i).name, 'original.jpg');
    
    % load image
    img = imread(imgfile);
    sz = size(img);
    
    y1 = round(2*sz(1)/5);
    y2 = round(3*sz(1)/5);
    x1 = round(2*sz(2)/5);
    x2 = round(3*sz(2)/5);
    % prepare initialization
    mask = zeros(sz);
    mask(y1:y2,x1:x2) = 1;
    
%     img = imresize(img,.5);
    mask = mask>0;
    
    % set input parameters
    lambda = .1;
    iterations = 1000;
    rad = 15;
    
    % perform segmentation
    [seg] = sfm_local_chanvese(img,mask,iterations,lambda,rad);
    SE=strel('disk',15);
    imekai=imerode(imdilate(seg*255,SE),SE);
    imwrite(uint8(imekai), fullfile(root_dir, folders(i).name, 'seg.jpg'))
    
    % display results
    % subplot(2,2,1)
    % imagesc(img); axis image; colormap gray;
    % title('The original image');
    %
    % subplot(2,2,2)
    % imagesc(mask); axis image; colormap gray;
    % title('The initialization');
    %
    % subplot(2,2,3)
    % imagesc(seg); axis image; colormap gray;
    % title('The final segmenatation output');
    %
    % subplot(2,2,4)
    % imagesc(img); axis image; colormap gray;
    % hold on;
    % contour(seg,[0 0],'r','linewidth',3);
    % hold off;
    % title('The image with the segmentation shown in red');
end

