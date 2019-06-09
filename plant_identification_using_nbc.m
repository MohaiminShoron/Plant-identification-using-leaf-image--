%adding different machine learning nad utility libraries.
addpath C:\Users\TheLazyProgrammer\utility
addpath C:\Users\TheLazyProgrammer\machineLearning
addpath C:\Users\TheLazyProgrammer\machineLearning\externalTool\libsvm-3.21\matlab
%getting the dataset and selecting 5 classes from there.
imDir='C:\Users\TheLazyProgrammer\utility\leafSorted';
opt=mmDataCollect('defaultOpt');
opt.extName='jpg';
opt.maxClassNum=5;
imageData=mmDataCollect(imDir, opt, 1);
% condition checking to make sure that the feature extraction process occurs only once.
opt=dsCreateFromMm('defaultOpt');
if exist('ds.mat', 'file')
	fprintf('Loading ds.mat...\n');
	load ds.mat
else
	myTic=tic;
	opt=dsCreateFromMm('defaultOpt');
	opt.imFeaFcn=@leafFeaExtract;	% Function for feature extraction
	opt.imFeaOpt=feval(opt.imFeaFcn, 'defaultOpt');	% Feature options
	ds=dsCreateFromMm(imageData, opt);
	fprintf('Time for feature extraction over %d images = %g sec\n', length(imageData), toc(myTic));
	fprintf('Saving ds.mat...\n');
	save ds ds
end
%various plots and charts are shown here regarding the features
figure; leafFeaExtract;
figure;
[classSize, classLabel]=dsClassSize(ds, 1);
figure; dsBoxPlot(ds);
figure; dsRangePlot(ds);
ds2=ds;
ds2.input=inputNormalize(ds2.input);
figure; dsFeaVecPlot(ds); figEnlarge;


%training using naive bayes classifier .
trainSet.input=ds2.input(:, 1:1:end); trainSet.output=ds2.output(:, 1:1:end);
 testSet.input=ds2.input(:, 1:2:end);  testSet.output=ds2.output(:, 2:2:end);
[cPrm, logLike1, recogRate1]=nbcTrain(trainSet);
[computedClass, logLike2, recogRate2, hitIndex]=nbcEval(testSet, cPrm, 1);
fprintf('Inside recog rate = %g%%\n', recogRate1*100);% accuracy if training set and testing set are the same.
fprintf('Outside recog rate = %g%%\n', recogRate2*100);% accuracy if training set and testing set are different.