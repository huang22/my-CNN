function mycnnTest(cnn,images,labels)
    numImages = length(images);
    activations = images;
    for l = 1:numel(cnn.layers)
        if(strcmp(cnn.layers{l}.type,'c'))%convolutional cnn.layers{l}
            activations = mycnnConvolve(activations,cnn.layers{l}.W,cnn.layers{l}.b);                    
        else
            activations = mycnnPool(cnn.layers{l}.poolDim,activations);
        end
        cnn.layers{l}.activations = activations;
    end
    %softmax
    activations = reshape(activations,[],numImages);
    probs = exp(bsxfun(@plus, cnn.Wd * activations, cnn.bd));
    sumProbs = sum(probs, 1);
    probs = bsxfun(@times, probs, 1 ./ sumProbs);
    
    [~,preds] = max(probs,[],1);
    preds = preds';
    
    acc = sum(preds==labels)/length(preds);
    fprintf('Accuracy is %f\n',acc);
end