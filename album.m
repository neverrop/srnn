function [A,ur] = album(features,A_sz,url,loop)

p = randperm(loop);
select = p(1:A_sz);
ur = cell(1,length(select));
A = cell(1,length(select));
for i = 1:length(select)
    t = (select(i):loop:size(features,1));
    A{i} = features(t,:);
    ur{i} = url(t);
end