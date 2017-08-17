function showSummaries(summary,path,lbpath,name,label_txt,user)

% summary = [2896;3075;3105;3184;3370;3395;3741;6630;8077;8375;8492;9186;10104];
% path = '/Users/never/data/GTEA_snack/en/Yin_Snack/';
% name = 'summary';

load(lbpath);
len = length(summary);
figure
h = tight_subplot(ceil(len/5),5, [0.01, 0.01], [0.01, 0.01], [0.01, 0.01]);
f = fopen(label_txt,'wt');
for i=1:len
%     subplot(len/5,5,i)
    I = imread(strcat(path,num2str(summary(i)),'.jpg'));
    axes(h(i));
    imshow(I);
    b = 'None';
    jj = 0;
    ll = 10000;
    for j=1:length(data)
        if (str2num(cell2mat(data{j,3}))<=summary(i)) && (summary(i)<=str2num(cell2mat(data{j,4})))
            %b = strcat(num2str(b),cell2mat(data{j,1}));\
            b = strcat(cell2mat(data{j,1}),' `',cell2mat(data{j,2}));
            continue
        end
        if min(abs(summary(i)-[str2num(cell2mat(data{j,3})),str2num(cell2mat(data{j,4}))]))<ll
            ll = min(abs(summary(i)-[str2num(cell2mat(data{j,3})),str2num(cell2mat(data{j,4}))]));
            jj = j;
        end
    end
    if strcmp(b,'None')
        b = strcat('0-',cell2mat(data{jj,1}),' `',cell2mat(data{jj,2})); 
    end
    title(b);
    drawnow
    line = strcat(num2str(summary(i)),' :',b);
    fprintf(f,'%s\n',line);
end
print(gcf,'-djpeg',strcat(path,name));
fclose(f);
end


function ha = tight_subplot(Nh, Nw, gap, marg_h, marg_w)


    if nargin<3; gap = .02; end
    if nargin<4 || isempty(marg_h); marg_h = .05; end
    if nargin<5; marg_w = .05; end

    if numel(gap)==1 
        gap = [gap gap];
    end
    if numel(marg_w)==1
        marg_w = [marg_w marg_w];
    end
    if numel(marg_h)==1
        marg_h = [marg_h marg_h];
    end

    axh = (1-sum(marg_h)-(Nh-1)*gap(1))/Nh; 
    %axh = axh/2;
    axw = (1-sum(marg_w)-(Nw-1)*gap(2))/Nw;

    py = 1-marg_h(2)-axh; 

    ha = zeros(Nh*Nw,1);
    ii = 0;
    for ih = 1:Nh
        px = marg_w(1);

        for ix = 1:Nw
            ii = ii+1;
            ha(ii) = axes('Units','normalized', ...
                'Position',[px py axw axh], ...
                'XTickLabel','', ...
                'YTickLabel','');
            px = px+axw+gap(2);
        end
        py = py-axh-gap(1);
    end
end
