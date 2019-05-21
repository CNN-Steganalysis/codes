function net = bias_initialization(net, imdb, len, index)

set = [ones(1,5000) 2*ones(1,5000)];
rnd_set = set(index);
cover_index = find(rnd_set == 1);
rnd_cover_index = randperm(length(cover_index));
batch = rnd_cover_index(1:len);

sample_path = strcat(imdb.coverDir, num2str(1), '.pgm');
cover = imread(sample_path);
im = single(zeros(size(cover,1),size(cover,2),1,2*length(batch)));

for i = 1 : length(batch)
   cover_path = strcat(imdb.coverDir, num2str(batch(i)), '.pgm'); 
   stego_path = strcat(imdb.stegoDir, num2str(batch(i)), '.pgm');
   cover = imread(cover_path);
   stego = imread(stego_path);
   
   im(:, :, 1, 2*i-1) = single(cover); 
   im(:, :, 1, 2*i) = single(stego);
end

inputs = {'data', im} ;
net.bias_init(inputs);

end