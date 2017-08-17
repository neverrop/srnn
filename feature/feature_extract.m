clear
close all


addpath /Users/never/caffe/matlab

% caffe.set_mode_cpu();
model= '/Users/never/caffe/models/bvlc_reference_caffenet/deploy.prototxt';
weights= '/Users/never/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel';
mean = load('/Users/never/caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat');
net = caffe.Net(model, weights, 'test'); % create net and load weights
mean_data = mean.mean_data;
  
net.blobs('data').reshape([227 227 3 1]);
net.reshape();

rt_img_dir='/Users/never/Documents/PycharmProjects/tubii/fc7/img1';
rt_data_dir='/Users/never/Documents/MATLAB/srnn-master/feature';



%disp('Extracting CNN features...');
        
        frames = dir(fullfile(rt_img_dir, '*'));    frames(1)=[];  frames(1)=[];      
        c_num = length(frames);           
        gray_num=0;error_num_CMYK_JPEG=0;
%        database.path=[];  
        
for jj = 1:c_num, imgpath = fullfile(rt_img_dir, frames(jj).name);
    try                    
        %% prepare the image
        im_data = caffe.io.load_image(imgpath);
        %% subtract mean_data (already in W x H x C, BGR)
        width = 256; height = 256;
        im_data = imresize(im_data, [width, height]); % resize using Matlab's imresize 
        feaSet.iscolor=1;
        if size(im_data,3)==1
            imdata=zeros([size(im_data),3]);
            imdata(:,:,1)=im_data;
            imdata(:,:,2)=im_data;
            imdata(:,:,3)=im_data;
            im_data=imdata;
            feaSet.iscolor=0;
            gray_num=gray_num+1;
        end
        im_data = im_data - (mean_data);   


        width = 227; height = 227;
        im_data = imresize(im_data, [width, height]); % resize using Matlab's imresize

        res = net.forward({im_data});
        fc6_data = net.blobs('fc6').get_data();
        fc7_data = net.blobs('fc7').get_data();


        feaSet.fc6_data = fc6_data;
        feaSet.fc7_data = fc7_data;

        [pdir, fname] = fileparts(frames(jj).name);                        
        fpath = fullfile(rt_data_dir, [fname, '.mat']);

        save(fpath, 'feaSet');
        %database.path = [database.path; fpath];
    catch
        str= fullfile(frames(jj).name);
        disp(str);
        error_num_CMYK_JPEG=error_num_CMYK_JPEG+1;
        error_CMYK_JPEG{error_num_CMYK_JPEG}=str;
    end
 end;    