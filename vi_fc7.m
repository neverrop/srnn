clear all
close all
% 
net = alexnet;
sz = net.Layers(1).InputSize;
layer = 'fc7';

path = '/Users/never/data/';
gzpath = '/Users/never/data/gaze/';
gtea = dir([path 'GTEA*']);

for t = 3:length(gtea)
    path1 = strcat(path,gtea(t).name,'/');
    videoPath = dir([path1 '/*.mp4']);
    disp(t)
    for j = 6:length(videoPath)
        tic;
        vi = VideoReader(strcat(path1,videoPath(j).name));
        load(strcat(gzpath,videoPath(j).name(1:(end-4)),'_gz.mat'));
        h = vi.Height;
        w = vi.Width;
        frame = length(gz)-40;
        features = [];
        features_crop = [];
        url = [];
        for i = 1:frame
            img = read(vi,gz(i,3));
            I_original = imresize(img,[sz(1),sz(2)]);
            x = round(min(max(gz(i,1),sz(1)),h-sz(1))) -sz(2);
            y = round(min(max(gz(i,2),sz(2)),h-sz(2))) -sz(1);
            I = imcrop(img,[y,x,2*sz(1)-1,2*sz(2)-1]);
            I = imresize(I,0.5);
            features = [features; activations(net,I_original,layer)];
            features_crop = [features_crop; activations(net,I,layer)];
            url = [url;gz(i,3)];
            img([(x+1):(x+10),(x+2*sz(2)-10):(x+2*sz(2))-1],(y+1):(y+2*sz(1))-1) = 225;
            img((x+1):(x+2*sz(2)-1),[(y+1):(y+10),(y+2*sz(1)-10):(y+2*sz(1)-1)]) = 225;
            img = imresize(img,0.5);
            imwrite(img,strcat(path1,videoPath(j).name(1:(end-4)),'/',num2str(gz(i,3)),'.jpg'),'jpg');
            %fr = [fr,gz(i,3)];
        end
        save(strcat(path1, videoPath(j).name(1:(end-4)) ,'_features.mat'),'features');
        features = features_crop;
        save(strcat(path1, videoPath(j).name(1:(end-4)) ,'_cropped_features.mat'),'features');
        save(strcat(path1, videoPath(j).name(1:(end-4)) ,'_url.mat'),'url');
        clear features img gz features_crop
        toc;
    end
end
