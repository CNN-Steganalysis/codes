function imdb= cnn_steganalysis_setup_data(cover_path, stego_path, mode)
% This function is to determine the training samples and testing samples

imdb.coverDir = cover_path;
imdb.stegoDir = stego_path;


% descriptions to the image database
imdb.meta.sets = {'train', 'val'};
imdb.meta.classes = {1,2};

% '1' represents the training sample, '2' represent the testing sample
if strcmp(mode, 'train')
    set = [ones(1,30000) 2*ones(1,10000)];
    index = randperm(length(set));
    imdb.images.set = set(index);
elseif strcmp(mode, 'test')
    set = [2*ones(1,30000) 2*ones(1,10000)];
    index = randperm(length(set));
    imdb.images.set = set(index);
else
    error('no corresponding operation');
end

end