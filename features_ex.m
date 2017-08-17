clear all
close all

net = alexnet
sz = net.Layers(1).InputSize

path = '/Users/never/data/GTEA_breakfast/';
imgDataDir = dir(path);
layer = 'fc7';

for i = 1:length(imgDataDir)
    tic;
    disp('working on'),i
    if(isequal(imgDataDir(i).name,'.')||... % ?????????????
       isequal(imgDataDir(i).name,'..')||...
       ~imgDataDir(i).isdir)                % ???????????
           continue;
    end
    features = [];
    imgDir = dir([path imgDataDir(i).name '/*.png']); 
    for j =1:length(imgDir)                 % ??????
        I = imread([path imgDataDir(i).name '/' imgDir(j).name]);
        I = imresize(I,[227,227]);
        features = [features; activations(net,I,layer)];
    end
    save(strcat(path, imgDataDir(i).name ,'/', imgDataDir(i).name ,'_features.mat'),'features');
    clear features
    toc
end



% files=dir('*.png');
% m=size(files,1);
% layer = 'fc7';
% features = []
% for i=1:m
%     I = imread(files(i).name);
%     I = I(1:sz(1),1:sz(2),1:sz(3));
%     features = [features; activations(net,I,layer)];
% end
% 
% save('features.mat',features)