function [ssc,ssc_q]=SSC_descriptor(I,sigma,delta)
% Calculation of SSC (self-similarity context)
%
% If you use this implementation please cite:
% M.P. Heinrich et al.: "Towards Realtime Multimodal Fusion for 
% Image-Guided Interventions Using Self-similarities"
% MICCAI (2013) LNCS Springer
%
% M.P. Heinrich et al.: "MIND: Modality Independent Neighbourhood
% Descriptor for Multi-Modal Deformable Registration"
% Medical Image Analysis (2012)
%
% Contact: heinrich(at)imi(dot)uni-luebeck(dot)de
%
% I: input volume (3D)
% sigma: Gaussian weighting for patches 
% delta: Distance between patch centres
%
% ssc: output descriptor (4D)
% ssc_q: quantised descriptor (3D) uint64
% important: ssc_q requires the compilitation of quantDescriptor.cpp

I=single(I);
if nargin<2
    sigma=0.8;
end
if nargin<3
    delta=1;
end

% Filter for efficient patch SSD calculation
filt=fspecial('gaussian',[ceil(sigma*3/2)*2+1,1],sigma);

%displacements between patches
dx=[+1,+1,-1,+0,+1,+0].*delta;
dy=[+1,-1,+0,-1,+0,+1].*delta;
dz=[+0,+0,+1,+1,+1,+1].*delta;

sx=[-1,+0,-1,+0,+0,+1,+0,+0,+0,-1,+0,+0].*delta;
sy=[+0,-1,+0,+1,+0,+0,+0,+1,+0,+0,+0,-1].*delta;
sz=[+0,+0,+0,+0,-1,+0,-1,+0,-1,+0,-1,+0].*delta;

% Self-similarity Distances
distances=zeros([size(I),numel(dx)],'single');

% Calculating Gaussian weighted patch SSD using convolution
for i=1:numel(dx)
    distances(:,:,:,i)=volfilter((I-volshift(I,dx(i),dy(i),dz(i))).^2,filt);
end

% Shift 'second half' of distances to avoid redundant calculations
index=[7,7,8,8,9,9,10,10,11,11,12,12]-6;
ssc=zeros([size(I),numel(index)],'single');
for i=1:numel(index)
    ssc(:,:,:,i)=volshift(distances(:,:,:,index(i)),sx(i),sy(i),sz(i));
end
clear distances;

% Remove minimal distance to scale descriptor to max=1
matrixmin=min(ssc,[],4);
for i=1:numel(index)
    ssc(:,:,:,i)=ssc(:,:,:,i)-matrixmin;
end

% Variance measure (standard absolute deviation)
V=mean(ssc,4);
val1=[0.001*(mean(V(:))),1000*mean(V(:))];
V=(min(max(V,min(val1)),max(val1)));

% descriptor calculation according
for i=1:numel(index)
    ssc(:,:,:,i)=exp(-ssc(:,:,:,i)./V);
end

if nargout>1
    % quantise descriptors into uint64
    ssc_q=quantDescriptor(ssc,6);

end

function vol=volfilter(vol,h,method)

if nargin<3
    method='replicate';
end
% volume filtering with 1D seperable kernel (faster than MATLAB function)
h=reshape(h,[numel(h),1,1]);
vol=imfilter(vol,h,method);
h=reshape(h,[1,numel(h),1]);
vol=imfilter(vol,h,method);
h=reshape(h,[1,1,numel(h)]);
vol=imfilter(vol,h,method);

function vol1shift=volshift(vol1,x,y,z)

[m,n,o,p]=size(vol1);

vol1shift=zeros(size(vol1));

x1s=max(1,x+1); x2s=min(n,n+x);
y1s=max(1,y+1); y2s=min(m,m+y);
z1s=max(1,z+1); z2s=min(o,o+z);

x1=max(1,-x+1); x2=min(n,n-x);
y1=max(1,-y+1); y2=min(m,m-y);
z1=max(1,-z+1); z2=min(o,o-z);

vol1shift(y1:y2,x1:x2,z1:z2,:)=vol1(y1s:y2s,x1s:x2s,z1s:z2s,:);
