opts.alpha = 1e-1;%learning
opts.batchsize = 100;
opts.numepochs = 1;
opts.imageDim = 28;
opts.imageChannel = 1;
opts.numClasses = 10;

opts.lambda = 0.0001; %weight decay
opts.momentum = .95;
opts.mom = 0.5;
opts.momIncrease = 20;

%Load Data
addpath ./common/;
images = loadMNISTImages('./common/train-images-idx3-ubyte');
images = reshape(images,opts.imageDim,opts.imageDim,1,[]);
labels = loadMNISTLabels('./common/train-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

testImages = loadMNISTImages('./common/t10k-images-idx3-ubyte');
testImages = reshape(testImages,opts.imageDim,opts.imageDim,1,[]);
testLabels = loadMNISTLabels('./common/t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10;

cnn.layers = {
    struct('type', 'c', 'numFilters', 6, 'filterDim', 5) %convolution layer
    struct('type', 's', 'poolDim', 2) %sub sampling layer
    struct('type', 'c', 'numFilters', 12, 'filterDim', 5) %convolution layer
    struct('type', 's', 'poolDim', 2) %subsampling layer
};

%build network
cnn = mycnnSetup(cnn,opts);
%train and test
mycnnTrain(cnn,images,labels,testImages,testLabels);