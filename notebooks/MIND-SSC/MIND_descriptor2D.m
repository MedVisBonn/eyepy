function mind=MIND_descriptor2D(I,r,sigma)

% Calculation of MIND (modality independent neighbourhood descriptor)
%
% If you use this implementation please cite:
% M.P. Heinrich et al.: "MIND: Modality Independent Neighbourhood
% Descriptor for Multi-Modal Deformable Registration"
% Medical Image Analysis (2012)
%
% Contact: mattias.heinrich(at)eng.ox.ac.uk
%
% I: input volume (2D)
% r: half-width of spatial search (large values may cause: "out of memory")
% r=0 uses a six-neighbourhood (Section 3.3) other dense sampling
% sigma: Gaussian weighting for patches (Section 3.1.1)
%
% mind: output descriptor (3D) (Section 3.1, Eq. 4)
%

I=single(I);
if nargin<3
    sigma=0.5;
end
if nargin<2
    r=0;
end

% Filter for efficient patch SSD calculation (see Eq. 6)
filt=fspecial('gaussian',[ceil(sigma*3/2)*2+1,ceil(sigma*3/2)*2+1],sigma);

[xs,ys]=search_region(r);

[xs0,ys0]=search_region(0);

Dp=zeros([size(I),length(xs0)],'single');

% Calculating Gaussian weighted patch SSD using convolution
for i=1:numel(xs0)
    Dp(:,:,i)=imfilter((I-imshift(I,xs0(i),ys0(i))).^2,filt);
end

% Variance measure for Gaussian function (see Section 3.2)
V=(mean(Dp,3));

% the following can improve robustness
% (by limiting V to be in smaller range)
val1=[0.001*(mean(V(:))),1000.*mean(V(:))];
V=(min(max(V,min(val1)),max(val1)));

I1=zeros([size(I),length(xs0)],'single');
for i=1:numel(xs0)
    I1(:,:,i)=exp(-Dp(:,:,i)./V);
end

mind=zeros([size(I),length(xs)],'single');
% descriptor calculation according to Eq. 4
if r>0
    for i=1:numel(xs)
        mind(:,:,i)=exp(-imfilter((I-imshift(I,xs(i),ys(i))).^2,filt)./V);
    end
else
    % if six-neighbourhood is used, all patch distances are already calculated
    for i=1:numel(xs0)
        mind(:,:,i)=I1(:,:,i);
    end
end
    

% normalise descriptors to a maximum/mean of 1 
max1=max(mind,[],3);
%max1=mean(mind,3);
for i=1:numel(xs)
    mind(:,:,i)=mind(:,:,i)./max1;
end

function [xs,ys]=search_region(r)

if r>0
    % dense sampling with half-width r
    [xs,ys]=meshgrid(-r:r,-r:r);
    xs=xs(:); ys=ys(:);
    mid=(length(xs)+1)/2;
    xs=xs([1:mid-1,mid+1:length(xs)]);
    ys=ys([1:mid-1,mid+1:length(ys)]);

else
    % six-neighbourhood
    xs=[1,-1,0,0];
    ys=[0,0,1,-1];

end

function im1shift=imshift(im1,x,y,pad)
if nargin<4
    pad=0;
end
[m,n,o]=size(im1);

im1shift=im1;
x1s=max(1,x+1);
x2s=min(n,n+x);

y1s=max(1,y+1);
y2s=min(m,m+y);

x1=max(1,-x+1);
x2=min(n,n-x);

y1=max(1,-y+1);
y2=min(m,m-y);

% length1=x2-x1+1;
% length2=x2s-x1s+1;
% length3=y2-y1+1;
% length4=y2s-y1s+1;


im1shift(y1:y2,x1:x2,:)=im1(y1s:y2s,x1s:x2s,:);

