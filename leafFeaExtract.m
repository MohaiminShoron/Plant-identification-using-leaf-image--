function fea=leafFeaExtract(im, opt, showPlot)
% feaExtract: Feature extraction for leaf images
%
%	Usage:
%		fea=leafFeaExtract(imFile, opt, showPlot)
%
%	Example:
%		imFile='C:\Users\TheLazyProgrammer\utility\leafSorted\Anhui Barberry\1552.jpg';
%		im=imread(imFile);
%		opt=leafFeaExtract('defaultOpt');
%		fea=leafFeaExtract(im, opt, 1);

%	Category: Leaf image feature extraction
%	Roger Jang, 20150110

if nargin<1, selfdemo; return; end
if ischar(im) && strcmpi(im, 'inputName')	% Return the input names
	fea={'a/p', 'eccentricity', 'major/minor', 'a/ca', 'mean', 'variance'};
	return
end
if ischar(im) && strcmpi(im, 'defaultOpt')	% Set the default options
	fea.dummyField=[];		% Dummy field to be added later
	return
end
if nargin<2||isempty(opt), opt=feval(mfilename, 'defaultOpt'); end
if nargin<3, showPlot=0; end

imGray=255-rgb2gray(im);
th=graythresh(imGray);		% Find the threshold by Otsu's method
imBw=im2bw(imGray, th);		% Separate the image into BW
props={'Area', 'Perimeter', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength', 'BoundingBox', 'ConvexArea', 'PixelIdxList'};
stats=regionprops(imBw, props);
[maxValue, maxIndex]=max([stats.Area]);
s=stats(maxIndex);		% Only consider the region with max. area
fea(1)=s.Area/s.Perimeter;
fea(2)=s.Eccentricity;
fea(3)=s.MajorAxisLength/s.MinorAxisLength;
fea(4)=s.Area/s.ConvexArea;
fea(5)=mean(imGray(s.PixelIdxList));
fea(6)=var(double(imGray(s.PixelIdxList)));
fea=fea(:);

if showPlot
	subplot(221);imshow(im); title('Original image');
	subplot(222); imshow(imGray); title('Image after rgb2gray()');
	subplot(223); imshow(imBw); title('Identified region(s)');
	for i=1:length(stats)
		boxOverlay(stats(i).BoundingBox, getColor(i), 1, int2str(i), 'bottom');
	end
	imTest=imGray;
	imTest(stats(maxIndex).PixelIdxList)=fea(5);
	subplot(224); imshow(imTest); title('Image after the major region being filled with its average gray level');
end

% ====== Self demo
function selfdemo
mObj=mFileParse(which(mfilename));
strEval(mObj.example);