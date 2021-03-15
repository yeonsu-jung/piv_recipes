%%
close all

img_path = '/Users/yeonsu/Dropbox (Harvard University)/Riblet/data/piv-data/2021-03-11/Flat_10 (black)_motor15/Flat_10 (black)_motor15.00_pos4_VOFFSET0_timing100_ag1_dg1_laser5_[03-11]/frame_000100.tiff';
im_raw = imread(img_path);
im = im_raw(:,430:end);

figure;imshow(im)
%%
