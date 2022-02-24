function [A,m,n] = sto_load(name, dir)
% load local data matrix 
% Input:
%       -- name : mnist / covtype
%       -- dir  : local director of file
% Output:
%       -- A : the data matrix
%       -- m : number of rows
%       -- n : number of cols

%% main
if name == "mnist"
    load(dir+'/mnist_uint8.mat'); 
    oddfilter = [1 0 1 0 1 0 1 0 1 0];
    test_y = 2*sum(double(test_y).*oddfilter,2)-1;
    train_y = 2*sum(double(train_y).*oddfilter,2)-1;
    A = [double(test_x).*test_y; double(train_x).*train_y];
    A = A./sqrt(dot(A',A'))';
    [m,n] = size(A);
else
    A = load(dir+'/covtype'); 
    A = A.covtype;
    [m,n] = size(A);
    for i=1:n-1
        A(:,i) = A(:,i)/norm(A(:,i));
        A(:,i) = A(:,i)-mean(A(:,i));
    end

    for i=1:m
        if(A(i,end)~=2)
            A(i,1:end-1) = -A(i,1:end-1);
        end
        A(i,1:end-1) = A(i,1:end-1)/norm(A(i,1:end-1));
    end

    A = A(:,1:end-1);
    [m,n] = size(A);
end

    
