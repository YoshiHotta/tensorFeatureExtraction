Deltas = [5 10 15];
kmax = 20;
% load MNIST data
Ntrain = 50000;
Nvarid = 10000;
Ntest = 10000;
classmap = containers.Map(0:9, 1:10);
[ Xtrain, Xvarid, Xtest, ytrain, yvarid, ytest ] = prepareMNIST( imgsize, Ntrain, Nvarid, Ntest, classmap );
XtrainResize = reshape(Xtrain, [Ntrain 28 28]);
XvaridResize = reshape(Xvarid, [Nvarid 28 28]);
XtestResize = reshape(Xtest, [Ntest 28 28]);

data = [];
for Delta = Deltas
    % choose orthogonal matrices
    T = tucker_als_feature(tensor(XtrainResize), [Ntrain Delta Delta], 'dimorder', [2 3 1]); % actually, the 1st index (corres. w/ 3) is not optimized
    U = T.U{2};
    V = T.U{3};
    % extract features
    featuresTrainReshape = ncon({T.core.data, T.U{1}}, {[1 -2 -3], [-1 1]});
    featuresVaridReshape = ncon({XvaridResize, U, V}, {[-1 1 2], [1 -2], [2 -3]});
    featuresTrain = reshape(featuresTrainReshape, [Ntrain Delta * Delta]);
    featuresVarid = reshape(featuresVaridReshape, [Nvarid Delta * Delta]);
    errorRateVarid_l = zeros(kmax, 1);
    for k = 1:kmax
        model = fitcknn(featuresTrain, ytrain, 'NumNeighbors',k,'Standardize',true);
        predVarid = predict(model, featuresVarid);
        errorRate = sum(predVarid ~= yvarid) / Nvarid;
        errorRateVarid_l(k) = errorRate;
    end
    % choose the best k
    [~, kbest] = min(errorRateVarid_l);
    
    % extract features of the test set
    featuresTestReshape = ncon({XtestResize, U, V}, {[-1 1 2], [1 -2], [2 -3]});
    featuresTest = reshape(featuresTestReshape, [Ntest Delta * Delta]);
    % calculate the error rate for the test data
    model = fitcknn(featuresTrain, ytrain, 'NumNeighbors',kbest,'Standardize',true);
    predTest = predict(model, featuresTest);
    errorRateTest = sum(predTest ~= ytest) / Ntest;
    data = [data; [kbest errorRateTest]];
end
save('../data/hosvdFeatureSelection2.mat', 'Deltas', 'kmax', 'Ntrain', 'Nvarid', 'Ntest', 'data');
