function mind=MIND_descriptor(I,r,sigma)

% Calculation of MIND (modality independent neighbourhood descriptor)
%
% If you use this implementation please cite:
% M.P. Heinrich et al.: "MIND: Modality Independent Neighbourhood
% Descriptor for Multi-Modal Deformable Registration"
% Medical Image Analysis (2012)
%
% Contact: mattias.heinrich(at)eng.ox.ac.uk
%
% I: input volume (3D)
% r: half-width of spatial search (large values may cause: "out of memory")
% r=0 uses a six-neighbourhood (Section 3.3) other dense sampling
% sigma: Gaussian weighting for patches (Section 3.1.1)
%
% mind: output descriptor (4D) (Section 3.1, Eq. 4)
%

I=single(I);
if nargin<3
    sigma=0.5;
end
if nargin<2
    r=0;
end

% Filter for efficient patch SSD calculation (see Eq. 6)
filt=fspecial('gaussian',[ceil(sigma*3/2)*2+1,1],sigma);

[xs,ys,zs]=search_region(r);

[xs0,ys0,zs0]=search_region(0);

Dp=zeros([size(I),length(xs0)],'single');

% Calculating Gaussian weighted patch SSD using convolution
for i=1:numel(xs0)
    Dp(:,:,:,i)=volfilter((I-volshift(I,xs0(i),ys0(i),zs0(i))).^2,filt);
end

% Variance measure for Gaussian function (see Section 3.2)
V=(mean(Dp,4));

% the following can improve robustness
% (by limiting V to be in smaller range)
val1=[sqrt(mean(V(:))),mean(V(:)).^2];
V=(min(max(V,min(val1)),max(val1)));

I1=zeros([size(I),length(xs0)],'single');
for i=1:numel(xs0)
    I1(:,:,:,i)=exp(-Dp(:,:,:,i)./V);
end

% normalise descriptors to a maximum of 1 (only within six-neighbourhood)
max1=max(I1,[],4);

mind=zeros([size(I),length(xs)],'single');
% descriptor calculation according to Eq. 4
if r>0
    for i=1:numel(xs)
        mind(:,:,:,i)=exp(-volfilter((I-volshift(I,xs(i),ys(i),zs(i))).^2,filt)./V)./max1;
    end
else
    % if six-neighbourhood is used, all patch distances are already calculated
    for i=1:numel(xs0)
        mind(:,:,:,i)=I1(:,:,:,i)./max1;
    end
end
    

function [xs,ys,zs]=search_region(r)

if r>0
    % dense sampling with half-width r
    [xs,ys,zs]=meshgrid(-r:r,-r:r,-r:r);
    xs=xs(:); ys=ys(:); zs=zs(:);
    mid=(length(xs)+1)/2;
    xs=xs([1:mid-1,mid+1:length(xs)]);
    ys=ys([1:mid-1,mid+1:length(ys)]);
    zs=zs([1:mid-1,mid+1:length(zs)]);
else
    % six-neighbourhood
    xs=[1,-1,0,0,0,0];
    ys=[0,0,1,-1,0,0];
    zs=[0,0,0,0,1,-1];
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
