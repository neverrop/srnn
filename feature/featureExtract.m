clear all
close all


addpath /Users/never/caffe/matlab
model= '/Users/never/caffe/models/bvlc_reference_caffenet/deploy.prototxt';
weights= '/Users/never/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel';
mean = load('/Users/never/caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat');
net = caffe.Net(model, weights, 'test'); % create net and load weights
mean_data = mean.mean_data;

net.blobs('data').reshape([227 227 3 1]);
net.reshape();

rt_img_dir='/Users/never/Documents/PycharmProjects/tubii/fc7/img1';
rt_data_dir='/Users/never/Documents/MATLAB/srnn-master/feature';

frames = dir(fullfile(rt_img_dir, '*'));    
frames(1)=[];  
c_num = length(frames);          
gray_num=0;
error_num_CMYK_JPEG=0;