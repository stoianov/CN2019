% [X,Y,N,A]=digitdata(n,sz)
% loads n digits of size sz for training a neural network classifier
%
% Example: 
% [X,Y,N,A]=digitdata(100,20);

function [X,Y,N,A]=digitdata(n,sz)
if nargin<2, sz=20; end
if nargin<1, n=100; end
% sz: 14,20,28
% n: number of samples per digit

switch sz
    case 28, fnm='Digits';
    case 20, fnm='Digits20x20';
    case 14, fnm='Digits14x14';
end
load(fnm);

X=[];
N=[];
A=[];

for i=0:9
  I=find(D.Num==i);
  nI=numel(I);
  I=I(randperm(nI,n));
  X=[X;D.IMG(I,:)];
  N=[N;D.Num(I,:)];
  A=[A;D.Ang(I,:)];
end

nn=numel(N);
Y=zeros(nn,10);         % memory for one-hot-coding
for i=1:nn
  Y(i,N(i)+1)=1;        % one-hot-coding of N (N is 0-9)
end

end
