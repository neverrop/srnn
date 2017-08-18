clear all;
pa = 'GTEA_snack';
%pa = 'GTEA_snack';
path = ['/Users/never/data/' pa '/'];

fprintf('Testing data from %s. \n',path);
featurePath = dir([path 'Yin*features.mat']);

for dpp = 1:2
    for j= 1:length(featurePath)
        summaries = cell(50);
        scores = nan(50,1);
        clear A
        name = featurePath(j).name;
        name1 = split(name,'_');
        user = strcat(char(name1(1)),'_',char(name1(2)));
        load(strcat(path,user,'_url.mat'));
        load(strcat(path,name));
        if name1(3) ~= 'cropped'
            load(strcat(path,pa,'_model.mat'));
            summary_name = '1summary';
            label_txt = strcat(path,user,'/','1lb.txt');
        else
            load(strcat(path,pa,'_cropped_model.mat'));
            summary_name = '1cropped_summary';
            label_txt = strcat(path,user,'/','1cropped_lb.txt');
        end
        [A,ur] = album(features,10,url);
        %if ~(matlabpool('size') > 0)
        %    matlabpool(6); 
        %end
        %parfor i = 1:length(A)
        for i = 1:length(A)
            url = ur{i};
            fprintf('%d\n',i);
            example = A{i};
            if size(example,1) < 20; continue; end;
            [outseq,outll,outseqind,outllall] = rnn_gen_album(net,example,dpp,'length',20+2,'genK',1000);
            best = outseqind{1};
            bestscore = outll(1);
            assert(best(end)-1 == length(url));
            indices = best(2:end-1)-1; %0 offset
            summaries{i} = url(best(2:end-1)+1-1); %inds should be one offset for urls, then back for [EOS album EOS]
            scores(i) = bestscore;
        end

        [~,i] = nanmax(scores);
        fprintf('best score: %d', i);
        bestsummary = summaries{i};
        if dpp==1
            showSummaries(bestsummary,strcat(path,user,'/'),strcat(path,'label/',user,'_notation.mat'),[summary_name '_dpp'],[label_txt,'_dpp'],user);
        else
            showSummaries(bestsummary,strcat(path,user,'/'),strcat(path,'label/',user,'_notation.mat'),summary_name,label_txt,user);
        end
    end
end