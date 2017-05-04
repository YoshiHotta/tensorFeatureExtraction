function [ Xtrain, Xvarid, Xtest, ytrain, yvarid, ytest ] = prepareMNIST( d, Ntrain, Nvarid, Ntest, classmap, varargin )
% MNISTを取得し、前処理を施す。
% 入力 : 
%   d : doubleの整数. イメージのピクセル数。元のピクセル数28^2より小さいときは画像を圧縮する。sqrt(28^2 / d)は整数でなければならない。
%   Ntrain : doubleの整数. 訓練データ数 (<= 60000)
%   Nvarid : doubleの整数. varidationデータ数 (<= 60000)
%   Ntest : doubleの整数. テストデータ数 (<= 10000)
%   classmap : containers.Map. クラス0,1,2, ...を1,2,3,...,10に変更する。
%   varargin (optional) : 長さ１の可変長引数。char. 画像圧縮のメソッド。デフォルトではnearest
% 出力 : 
%   Xtrain : Ntrain * dのdoubleの行列。 
%   Xvarid : Nvarid * dのdoubleの行列。 
%   Xtest : Ntest * dのdoubleの行列。 
%   ytrain : Ntrain * 1のdoubleの行列。
%   yvarid :Nvarid * 1のdoubleの行列。
%   ytest : Ntest * 1のdoubleの行列。

IsInteger = @(x) (abs(round(x)-x)) <= eps('double');
if  Ntrain + Nvarid > 60000 || Ntest > 10000
    error('ArgCheck:ArgRangeError', 'データ数の指定が間違っています。');
end

if ~IsInteger( sqrt(28^2/d) ) ||  d <= 0 || d > 28^2
    error('ArgCheck:dValueError', 'sqrt( 28^2 / d )は整数でなければいけません。');
end

if length(varargin) == 0
    method = 'nearest';
elseif length(varargin) == 1
    method = varargin{1} ;
else
    error('ArgCheck:VararginError', '可変長引数が長すぎます。');
end

[XtrainPlusVarid, yTrainPlusVaridRaw, Xtest, yTestRaw] = setupMnist('binary', false,'keepSparse', false);
XtrainPlusVarid = XtrainPlusVarid(1:Ntrain + Nvarid, :);
yTrainPlusVaridRaw = yTrainPlusVaridRaw(1:Ntrain + Nvarid, :);
Xtest = Xtest(1:Ntest, :);
yTestRaw = yTestRaw(1:Ntest, :);

yTrainPlusVarid = zeros(Ntrain + Nvarid, 1);
for n = 1:Ntrain + Nvarid
    yTrainPlusVarid(n) = classmap(yTrainPlusVaridRaw(n));
end
ytest = zeros(Ntest, 1);
for n = 1:Ntest
    ytest(n) = classmap(yTestRaw(n));
end

order1 = randperm(size(yTrainPlusVarid,1));
XtrainPlusVaridRand = XtrainPlusVarid(order1, :); 
yTrainPlusVaridRand = yTrainPlusVarid(order1, :);
Xtrain = XtrainPlusVaridRand(1:Ntrain,:);
Xvarid = XtrainPlusVaridRand(Ntrain+1:end, :);
ytrain = yTrainPlusVaridRand(1:Ntrain, :);
yvarid = yTrainPlusVaridRand(Ntrain+1:end, :);
order2 = randperm(size(ytest,1));
Xtest = Xtest(order2, :);
ytest = ytest(order2, :);



cprRate = 1 / sqrt(28^2 / d);
if cprRate ~= 1
    XtrainSmall = zeros(Ntrain, d);
    XvaridSmall = zeros(Nvarid, d);
    XtestSmall = zeros(Ntest, d);
    for n = 1:Ntrain
        x = Xtrain(n, :);
        x = reshape(x, [28 28]);
        xsmall = imresize(x, cprRate, method);
        xsmall = reshape(xsmall, [1 d]);
        XtrainSmall(n, :) = xsmall;
    end
    for n = 1:Nvarid
        x = Xvarid(n, :);
        x = reshape(x, [28 28]);
        xsmall = imresize(x, cprRate, method);
        xsmall = reshape(xsmall, [1 d]);
        XvaridSmall(n, :) = xsmall;
    end
    for n = 1:Ntest
        x = Xtest(n, :);
        x = reshape(x, [28 28]);
        xsmall = imresize(x, cprRate, method);
        xsmall = reshape(xsmall, [1 d]);
        XtestSmall(n, :) = xsmall;
    end
    Xtrain = XtrainSmall;
    Xvarid = XvaridSmall;
    Xtest = XtestSmall;
end

Xtrain = Xtrain / 255;
Xvarid = Xvarid / 255;
Xtest = Xtest / 255;
end