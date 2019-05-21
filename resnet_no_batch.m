function net = resnet_no_batch(varargin)

opts.batchNormalization = true; 
opts.networkType = 'resnet'; % 'plain' | 'resnet'
opts.bottleneck = false; % only used when n is an array
opts.nClasses = 2;
opts.reLUafterSum = false;
opts = vl_argparse(opts, varargin); 
nClasses = opts.nClasses;


net = dagnn.DagNN();

% Meta parameters
net.meta.inputSize = [512 512 1] ;
net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.trainOpts.momentum = 0.9;
net.meta.trainOpts.batchSize = 20 ;

net.meta.trainOpts.learningRate = [0.01*ones(1,142) 0.001*ones(1,50)] ;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% add a filter layer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% layer 0: preprocessing layer
block = dagnn.Conv('size',  [5 5 1 20], 'hasBias', true, ...
           'stride', 1, 'pad', [2 2 2 2]);
lName = 'conv0';
net.addLayer(lName, block, 'data', lName, {[lName '_f'] [lName '_b']});
net.params(1).learningRate = 0.00001;
block = dagnn.Trunc();
net.addLayer('trunc', block, 'conv0', 'trunc');

% layer 1
block = dagnn.Conv('size',  [3 3 20 24], 'hasBias', true, ...
           'stride', 1, 'pad', 1);
lName = 'conv1';
net.addLayer(lName, block, 'trunc', lName, {[lName '_f'] [lName '_b']});
block = dagnn.ReLU('leak',0);
net.addLayer('relu1',  block, 'conv1', 'relu1');


% layer 2
block = dagnn.Conv('size',  [3 3 24 24], 'hasBias', true, ...
           'stride', 1, 'pad', 1);
lName = 'conv2';
net.addLayer(lName, block, 'relu1', lName, {[lName '_f'] [lName '_b']});
block = dagnn.ReLU('leak',0);
net.addLayer('relu2',  block, 'conv2', 'relu2');
block = dagnn.Pooling('poolSize', [3 3], 'method', 'avg', 'pad', [0 1 0 1], 'stride', 2); 
net.addLayer('pool2', block, 'relu2', 'pool2');

% layer 3
block = dagnn.Conv('size',  [3 3 24 48], 'hasBias', true, ...
           'stride', 1, 'pad', 1);
lName = 'conv3';
net.addLayer(lName, block, 'pool2', lName, {[lName '_f'] [lName '_b']});
block = dagnn.ReLU('leak',0);
net.addLayer('relu3',  block, 'conv3', 'relu3');
block = dagnn.Pooling('poolSize', [3 3], 'method', 'avg', 'pad', [0 1 0 1], 'stride', 2); 
net.addLayer('pool3', block, 'relu3', 'pool3');

% layer 4
block = dagnn.Conv('size',  [3 3 48 96], 'hasBias', true, ...
           'stride', 1, 'pad', 1);
lName = 'conv4';
net.addLayer(lName, block, 'pool3', lName, {[lName '_f'] [lName '_b']});
block = dagnn.ReLU('leak',0);
net.addLayer('relu4',  block, 'conv4', 'relu4');
block = dagnn.Pooling('poolSize', [3 3], 'method', 'avg', 'pad', [0 1 0 1], 'stride', 2); 
net.addLayer('pool4', block, 'relu4', 'pool4');

% layer 5
block = dagnn.Conv('size',  [3 3 96 192], 'hasBias', true, ...
           'stride', 1, 'pad', 1);
lName = 'conv5';
net.addLayer(lName, block, 'pool4', lName, {[lName '_f'] [lName '_b']});
block = dagnn.ReLU('leak',0);
net.addLayer('relu5',  block, 'conv5', 'relu5');
block = dagnn.Pooling('poolSize', [3 3], 'method', 'avg', 'pad', [0 1 0 1], 'stride', 2); 
net.addLayer('pool5', block, 'relu5', 'pool5');


% appended layers
block = dagnn.Pooling('poolSize', [32 32], 'method', 'avg', 'pad', 0, 'stride', 1);
net.addLayer('pool_final', block, 'pool5', 'pool_final');
block = dagnn.Conv('size', [1 1 192 nClasses], 'hasBias', true, ...
                   'stride', 1, 'pad', 0);
lName = 'fc_label';
net.addLayer(lName, block, 'pool_final', lName, {[lName '_f'], [lName '_b']});

net.addLayer('softmax', dagnn.SoftMax(), lName, 'softmax');  
net.addLayer('loss', dagnn.Loss('loss', 'log'), {'softmax', 'label'}, 'loss');
net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'softmax','label'}, 'error') ;
net.initParams();

net.meta.normalization.imageSize = net.meta.inputSize;
net.meta.normalization.border = 512 - net.meta.inputSize(1:2) ;
net.meta.normalization.interpolation = 'bicubic' ;
net.meta.normalization.averageImage = [] ;
net.meta.normalization.keepAspect = true ;
net.meta.augmentation.rgbVariance = zeros(0,3) ;
net.meta.augmentation.transformation = 'stretch' ;



