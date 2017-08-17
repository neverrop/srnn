for i = 1:10
    idxcls = sprintf('%d', i)
    subplot(2,5,i)
    imgname = [idxcls, '.jpg']
    img  = imread(imgname)
    imshow(img)
end

saveas(gcf,'paris.png')