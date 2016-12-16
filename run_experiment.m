clear all;  close all;
addpath(genpath('dependencies\')); 

setup; % set up the data dependencies
opts.imdb       = [];
opts.networkType = 'resnet' ; 
opts.expDir = 'results'; % the name of folder to save training results
opts.nClasses = 2;
opts.batchSize = 10; % the size of batch
opts.numAugments = 1 ;
opts.numEpochs = 200; % number of training epoches
opts.gpus = [];  % the ID of gpu for model training or testing 
opts.checkpointFn = [];

% The path of cover images and stego images, you can change this path
cover_path = 'F:\BOSS\crop_BOSS\cover\';
stego_path = 'F:\BOSS\crop_BOSS\suniward_04\';

% -------------------------------------------------------------------------
%                                    test the proposed model
% -------------------------------------------------------------------------
% fileName = '..\nets\trained_net.mat'; % load a well trained model for validation 
% load(fileName, 'net', 'stats') ;
% net = dagnn.DagNN.loadobj(net) ;
% imdb = cnn_steganalysis_setup_data(cover_path, stego_path,'test'); % determine the training data and testing data
% 
% % model testing
% testfn = @test_model;
% [net, info] = testfn(net, imdb, getBatchFn(opts, net.meta), ...
%   'expDir', opts.expDir, ...
%   net.meta.trainOpts, ...
%   'gpus', opts.gpus, ...
%   'batchSize',opts.batchSize,...
%   'numEpochs',opts.numEpochs,...
%   'val', find(imdb.images.set == 2), ...
%   'derOutputs', {'loss', 1}, ...
%   'checkpointFn', opts.checkpointFn) ;

% -------------------------------------------------------------------------
%                                    train the proposed model
% -------------------------------------------------------------------------

payload = 0.4; % the payload size
fileName = '..\nets\untrained_net.mat'; % load an untrained model
load(fileName, 'net', 'stats') ;
net = dagnn.DagNN.loadobj(net) ;
imdb = cnn_steganalysis_setup_data(cover_path, stego_path, 'train');

%model training
trainfn = @train_model;
[net, info] = trainfn(net, imdb, getBatchFn(opts, net.meta), payload, ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  'gpus', opts.gpus, ...
  'batchSize',opts.batchSize,...
  'numEpochs',opts.numEpochs,...
  'val', find(imdb.images.set == 2), ...
  'derOutputs', {'loss', 1}, ...
  'checkpointFn', opts.checkpointFn) ;

