clear all
close all

path = '/Users/never/data/';
gtea = dir([path 'GTEA*']);

for t = 1:length(gtea)
    path1 = strcat(path,gtea(t).name,'/');
    for c = 1:2
        if c==1
            cc='cropped_';
        else
            cc='';
        end
        fePath1 = strcat(path1,'*_',cc,'features.mat');
        %a = ['*_',cc,'features.mat'];
        fePath = dir(fePath1);
        fe = {};
        for j = 1:length(fePath)
            load(strcat(path1,fePath(j).name));
            fe = [fe , features];
        end
        save(strcat(path, 'all_',cc,'fe.mat'),'fe');
    end
end