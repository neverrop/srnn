clear all;
dbstop if error;
testdata = {'GTEA_American','GTEA_Pizza','GTEA_Snack'};
subsampl = 15 + 2;

for pp = 3:length(testdata)
    pa = char(testdata(pp));
    pa1 = pa;
    path = ['/home/never/data/' pa '/'];
    fprintf('Testing data from %s. \n',path);
    %featurePath = dir([path '*features.mat']);
    featurePath = dir([path '*_features.mat']);

    for dpp = 2
        for j= 1:length(featurePath)
            summaries = cell(50);
            scores = nan(50,1);
            clear A
            name = featurePath(j).name;
            name1 = split(name,'_');
            user = strcat(char(name1(1)),'_',char(name1(2)));
            load(strcat(path,user,'_url.mat'));
            load(strcat(path,name));
            if ~strcmp(char(name1(3)),'cropped')
                load(strcat(path,pa1,'_',num2str(subsampl-2),'_model.mat'));
                summary_name = '1summary';
                label_txt = strcat(path,user,'/','_1lb.txt');
            else
                load(strcat(path,pa1,'_cropped_',num2str(subsampl-2),'_model.mat'));
                summary_name = '1cropped_summary';
                label_txt = strcat(path,user,'/','_1cropped_lb.txt');
            end
            fe = {};
            ur = {};
            se = ceil(size(features,1)/10000);
            [A1,ur1] = album(features,24,url,24*se);
            [A2,ur2] = album(features,24,url,48*se);
            for jjj = 1:length(A1)
                fe = [fe,A1{jjj},A2{jjj}];
                ur = [ur,ur1{jjj},ur2{jjj}];
            end
            %[A,ur] = album(features,12,url);
            %if ~(matlabpool('size') > 0)
            %    matlabpool(6); 
            %end
            %parfor i = 1:length(A)
            A = fe;
            for i = 1:length(A)
                url = ur{i};
                fprintf('%d\n',i);
                example = A{i};
                if size(example,1) < 20; continue; end
                [outseq,outll,outseqind,outllall] = rnn_gen_album(net,example,dpp,'length',subsampl,'genK',1000);
                best = outseqind{1};
                bestscore = outll(1);
                assert(best(end)-1 == length(url));
                indices = best(2:end-1)-1; %0 offset
                %a = best(2:end-1)-1;
                %a(find(a==0))= [1]; %---------------------------->
                summaries{i} = urls(best(2:end-1)+1-1); %inds should be one offset for urls, then back for [EOS album EOS]
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
end