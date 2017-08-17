path = '/Users/never/data/GTEA_Pizza/label/';
files =dir(strcat(path,'*.txt'));

for i=1:length(files)
    f = fopen(strcat(path,files(i).name));
    data = {};
    j = 1;
    while ~feof(f)
        c = {};
        l = fgetl(f);
        lb = strsplit(l,{'>','<','(',')','-'});
        data{j,1} = strcat(lb(2));
        data{j,2} = strcat(lb(3));
        data{j,3} = lb(5);
        data{j,4} = lb(6);
        j = j+1;
    end
    save(strcat(path,files(i).name(1:(end-4)),'_notation.mat'),'data');
end