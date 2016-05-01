function pooledFeatures = mycnnPool(poolDim, convolvedFeatures)
	numImages = size(convolvedFeatures, 4);
	numFeatures = size(convolvedFeatures, 3);
	convolvedDim = size(convolvedFeatures, 1);

	pooledFeatures = zeros(floor(convolvedDim / poolDim), floor(convolvedDim / poolDim),numFeatures, numImages);

	for imageNum = 1:numImages
    	for featureNum = 1:numFeatures
        	featuremap = squeeze(convolvedFeatures(:,:,featureNum,imageNum));
        	pooledFeaturemap = conv2(featuremap,ones(poolDim)/(poolDim^2),'valid');
        	pooledFeatures(:,:,featureNum,imageNum) = pooledFeaturemap(1:poolDim:end,1:poolDim:end);
    	end
	end
end
