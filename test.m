clear all
close all

outputname = 'output.mat';
outputnamebest = 'outputbest.mat';
summaries = cell(50);
scores = nan(50,1);
clear A
load('/Users/never/data/GTEA_snack/en/snack_model.mat');
%path = '/Users/never/data/GTEA_breakfast/Ahmad_American/';

path = '/Users/never/data/GTEA_snack/en/';
load(strcat(path, 'Yin_Snack_features.mat'));
A = features;

    
%if ~(matlabpool('size') > 0)
%    matlabpool(6); 
%end
%parfor i = 1:length(A)
example = A;
% if size(example,1) < 20; continue; end;
url = linspace(1,size(A,1),size(A,1));
[outseq,outll,outseqind,outllall] = rnn_gen_album(net,example,'length',10+2,'genK',1000);
best = outseqind{1};
bestscore = outll(1);
assert(best(end)-1 == length(url));
indices = best(2:end-1)-1; %0 offset
summaries = url(best(2:end-1)+1-1); %inds should be one offset for urls, then back for [EOS album EOS]
scores = bestscore;

save(outputname,'summaries');
% [~,1] = nanmax(scores);
% fprintf('best score: %d', 1);
bestsummary = summaries;
save(outputnamebest,'bestsummary'),

showSummaries(bestsummary,strcat(path,'Yin_Snack/'),strcat(path,'summary.jpeg'));