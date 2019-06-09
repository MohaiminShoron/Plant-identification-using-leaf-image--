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

%training using K-nearest neighbor classifier .
rr=knncLoo(ds);
fprintf('rr=%g%% for ds\n', rr*100);
[rr, computed]=knncLoo(ds2);
fprintf('rr=%g%% for ds2 of normalized inputs\n', rr*100);

