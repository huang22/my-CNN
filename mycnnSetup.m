function cnn = mycnnSetup(cnn,opts)
    outDim = opts.imageDim;
    
    for l = 1:numel(cnn.layers)
        if(strcmp(cnn.layers{l}.type,'c'))%convolutional
           numFilters = cnn.layers{l}.numFilters;
           filterDim = cnn.layers{l}.filterDim;

           %initialize
           cnn.layers{l}.W = 1e-1*randn(filterDim,filterDim,opts.imageChannel,numFilters);
           cnn.layers{l}.b = zeros(numFilters,1);
           cnn.layers{l}.W_velocity = zeros(size(cnn.layers{l}.W));
           cnn.layers{l}.b_velocity = zeros(size(cnn.layers{l}.b));
           
           %after convolution
           convDim = outDim - cnn.layers{l}.filterDim + 1;
           cnn.layers{l}.delta = zeros(convDim,convDim,numFilters,opts.batchsize);

           outDim = convDim;
        else%pooling
          %after pooling
           pooledDim = outDim / cnn.layers{l}.poolDim;
           cnn.layers{l}.delta = zeros(pooledDim,pooledDim,opts.imageChannel,opts.batchsize);
           outDim = pooledDim;
        end

    end
    
    cnn.hiddenSize = outDim ^ 2 * numFilters;

    cnn.probs = zeros(opts.numClasses,opts.batchsize);
    
    r  = sqrt(6) / sqrt(opts.numClasses+cnn.hiddenSize+1);
    cnn.Wd = rand(opts.numClasses, cnn.hiddenSize) * 2 * r - r;
    cnn.bd = zeros(opts.numClasses,1);
    cnn.Wd_velocity = zeros(size(cnn.Wd));
    cnn.bd_velocity = zeros(size(cnn.bd));
    cnn.delta = zeros(size(cnn.probs)); 
    
    cnn.cost = 0;
    cnn.imageDim = opts.imageDim;
    cnn.imageChannel = opts.imageChannel;
    cnn.numClasses = opts.numClasses;
    cnn.alpha = opts.alpha; %learning rate
    cnn.minibatch = opts.batchsize;
    cnn.numepochs = opts.numepochs;
    cnn.lambda = opts.lambda;
    cnn.momentum = opts.momentum;
    cnn.mom = opts.mom;
    cnn.momIncrease = opts.momIncrease;
end