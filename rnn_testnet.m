%
% Generate summaries for multiple photo albums using the trained SRNN
% Calls RNN_GEN_ALBUM to generate a likely sequence using the SRNN which is the summary
%
% Gunnar Atli Sigurdsson & Xinlei Chen 2015
% Carnegie Mellon University
clear all;
path = '/Users/never/data/GTEA_snack/en/';
outputname = 'output.mat';
outputnamebest = 'outputbest.mat';
summaries = cell(50);
scores = nan(50,1);
clear A
load('/Users/never/data/GTEA_snack/Yin_Snack_url.mat');

% load('/Users/never/data/GTEA_snack/en/snack_model.mat');
% load('/Users/never/data/GTEA_snack/en/Yin_Snack_features.mat');
% A{1} = features;
% A{2} = features;
% A{3} = features;
% A{4} = features;

load('/Users/never/data/GTEA_snack/cropped_snack_model.mat');
load('/Users/never/data/GTEA_snack/Yin_Snack_cropped_features.mat');
A{1} = features_crop;
A{2} = features_crop;
A{3} = features_crop;
A{4} = features_crop;
    
%if ~(matlabpool('size') > 0)
%    matlabpool(6); 
%end
%parfor i = 1:length(A)
for i = 1:length(A)
    fprintf('%d\n',i);
    example = A{i};
    if size(example,1) < 20; continue; end;
    [outseq,outll,outseqind,outllall] = rnn_gen_album(net,example,'length',15+2,'genK',1000);
    best = outseqind{1};
    bestscore = outll(1);
    assert(best(end)-1 == length(url));
    indices = best(2:end-1)-1; %0 offset
    summaries{i} = url(best(2:end-1)+1-1); %inds should be one offset for urls, then back for [EOS album EOS]
    scores(i) = bestscore;
end

save(outputname,'summaries');
[~,i] = nanmax(scores);
fprintf('best score: %d', i);
bestsummary = summaries{i};
save(outputnamebest,'bestsummary'),

showSummaries(bestsummary,'/Users/never/data/GTEA_snack/en/Yin_Snack/','1cropped_summary');
%showSummaries(bestsummary,'/Users/never/data/GTEA_snack/en/Yin_Snack/','summary');

