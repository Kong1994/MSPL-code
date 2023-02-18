clear
addpath(genpath(cd))

load video1.mat
X=video1;
%X=X/255;
maxP = max(abs(X(:)));
[n1,n2,n3] = size(X);
opts.mu = 1e-4;
opts.mu_bar = 1e10;
opts.tol = 1e-6;
opts.rho = 1.1;
opts.maxIter = 500;
opts.DEBUG =1;
 p =0.1;    
%% ���������ӿռ���Ϣ
%% trainin
tl=60;
DataTrain=(X(:,:,1:tl));
XT=X(:,:,tl+1:100);
shift_dim=[1 3 2];
DataTrain=permute(DataTrain,shift_dim);
XT=permute(XT,shift_dim);
[d1,d2,d3]=size(DataTrain);
[k1,k2,k3] = size(XT);
[Phat,Lamhat,Rhat] = tsvd((DataTrain));
T=energythreshold(diag(Lamhat(:,:,1)),0.9999); %% 0.99 for Yale
rhat=length(T);
U01=Phat(:,T,:);
V01=Rhat(1:k2,T,:); 
t1=[1 2 3 4 5];
U02=Phat(:,T(t1),:);
V02=Rhat(1:k2,T(t1),:); 
t2=[2 3 1];
U03=Phat(:,T(t2),:);
V03=Rhat(1:k2,T(t2),:); 
A1=teye(d1,d3)-tprod(U01,tran(U01));
B1=teye(k2,k3)-tprod(V01,tran(V01));
A2=teye(d1,d3)-tprod(U02,tran(U02));
B2=teye(k2,k3)-tprod(V02,tran(V02));
A3=teye(d1,d3)-tprod(U03,tran(U03));
B3=teye(k2,k3)-tprod(V03,tran(V03));
%%  Set  Omega0.1
fprintf('Sampling ratio = %0.8e\n',p);
temp = randperm(k1*k2*k3);
kks = round(p*k1*k2*k3);
omega=temp(1:kks);
mark = zeros(k1,k2,k3); 
mark(temp(1:kks)) = 1;
M=XT.*mark;
tic
[Xhat_M,err] =lrtc_tnnm3(M,omega ,A1,B1,A2,B2,A3,B3,opts);
%[Xhat_M,err] =lrtc_tnnm2(M,omega ,A1,B1,A2,B2,opts);% for YaleB
toc
M=permute(M,shift_dim);
XT=permute(XT,shift_dim);
Xhat_M=permute(Xhat_M,shift_dim);
Xhat_M = max(Xhat_M,0);
Xhat_M = min(Xhat_M,maxP);
%% print the relative error, psnr,ssim
psnr_M = PSNR(XT,Xhat_M,maxP);
ssim_M = ssim(Xhat_M,XT);
fsim_M=FeatureSIM(Xhat_M,XT);
ErrorsM= norm(Xhat_M(:)-XT(:))/norm(XT(:));
fprintf('Relative error = %0.8e\n',ErrorsM);
D=Xhat_M(:)-XT(:);
MSE_M = sum(D(:).*D(:))/numel(Xhat_M(:));
MAE_M=mean(mean(abs(D)));%ƽ���������
fprintf('psnr_n1 = %0.2f  ssim_nl = %0.4f  fsim_nl = %0.4f MSE = %0.4f MAE = %0.4f ',psnr_M,ssim_M,fsim_M,MSE_M,MAE_M); 







