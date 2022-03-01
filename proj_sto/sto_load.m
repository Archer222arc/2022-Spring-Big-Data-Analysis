function [A_train,A_test,m,n] = sto_load(name, dir)
% load local data matrix 
% Input:
%       -- name : mnist / covtype
%       -- dir  : local director of file
% Output:
%       -- A : the data matrix
%       -- m : number of rows
%       -- n : number of cols

%% main
% if name == "mnist"
%     load(dir+'/mnist_uint8.mat'); 
%     oddfilter = [1 0 1 0 1 0 1 0 1 0];
%     test_y = 2*sum(double(test_y).*oddfilter,2)-1;
%     train_y = 2*sum(double(train_y).*oddfilter,2)-1;
%     A = [double(test_x).*test_y; double(train_x).*train_y];
%     A = A./sqrt(dot(A',A'))';
%     [m,n] = size(A);
    
if name =='covtype'
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
        A(i,1:end-1) = A(i,1:end-1)/(norm(A(i,1:end-1))+eps);
    end
    t = randperm(m,floor(m*0.7));
    
    A_train = A(t,1:end-1);
    A_test = A(setdiff(1:m,t),1:end-1);
    [m,n] = size(A);
else
    A_train = load(dir+'/gisette_train');
    A_train = A_train.A_train;
    A_test  = load(dir+'/gisette_test');
    A_test = A_test.A_test;
    y_train = load(dir+'/gisette_train_y');
    y_train = y_train.y_train;
    y_test = load(dir+'/gisette_test_y');
    y_test = y_test.y_test;
    for i = 1:5000
        A_train(:,i) = A_train(:,i)/(norm(A_train(:,i))+eps);
        A_test(:,i) = A_test(:,i)/(norm(A_test(:,i))+eps);
    end
    for i = 1:6000
        A_train(i,:) = A_train(i,:)*y_train(i);
    end
    for i = 1:1000
        A_test(i,:) = A_test(i,:)*y_test(i);
    end
    [m,n] = size(A_train);
end

    
