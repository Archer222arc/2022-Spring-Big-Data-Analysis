A1 = imread('image.jpeg'); %read the image peppers.png
% imshow(A1); %display the image
A = rgb2gray(A1); %Convert to grayscale
imwrite(A,"./fig/image_gray.jpeg");
A = double(A); %convert the type of data to double
% imshow(uint8(A));

[Ut,St,Vt] = svd(A);
St = diag(St);
Anorm = norm(A,'F');
Unorm = norm(Ut,'F');
Vnorm = norm(Vt,'F');
Snorm = norm(St,'F');
opts = [];
k_list = [5,10,15,20,100,200];
data = zeros(3,5);

for i = 1:6
    k =k_list(i);
tic;
[U, V, d] = svd_Lineartime(A,k,10*k,opts);
data(1,1) = toc;
data(1,2) = norm(d-St(1:k),'fro')/Snorm;
data(1,3) = compare_dist(U,Ut(:,1:k))/Unorm;
data(1,4) = compare_dist(V,Vt(:,1:k))/Vnorm;
As = U*diag(d)*V';
data(1,5) = norm(As-A,'F')/Anorm;
imwrite(uint8(As),"./fig/image_rec"+string(i)+string(1)+".jpeg");

[U, V, d] = svd_prototype(A,k,1,opts);
data(2,1) = toc;
data(2,2) = norm(d-St(1:k),'fro')/Snorm;
data(2,3) = compare_dist(U,Ut(:,1:k))/Unorm;
data(2,4) = compare_dist(V,Vt(:,1:k))/Vnorm;
As = U*diag(d)*V';
data(2,5) = norm(As-A,'F')/Anorm;
imwrite(uint8(As),"./fig/image_rec"+string(i)+string(2)+".jpeg");

tic;
[U, S,V] = svds(A,k);
d = diag(S);
data(3,1) = toc;
data(3,2) = norm(d-St(1:k),'fro')/Snorm;
data(3,3) = compare_dist(U,Ut(:,1:k))/Unorm;
data(3,4) = compare_dist(V,Vt(:,1:k))/Vnorm;
As = U*diag(d)*V';
data(3,5) = norm(As-A,'F')/Anorm;
imwrite(uint8(As),"./fig/image_rec"+string(i)+string(3)+".jpeg");

label.title1 = ["Algorithm","CPU time","Error singular value","Error $U$","Error $V$","Error $A$"];
label.length1 = [1,1,1,1,1];
% label.title2 = ["-","$c=2k$","$c=10k$","$c=50k$","$q=0$","$q=1$","$q=2$"];
label.col = ["Linear time","Prototype","SVDS (baseline)"];
opt.H = "None";

% time
opt.caption = "Comparison of proposed algorithms on realistic image with baseline, $k="+string(k)+"$";
opt.label = "image"+string(i);
opt.filename = "./table/image"+string(i);
maketable(data,label,opt);
end

function dis = compare_dist(U1,U2)

[~,k] = size(U1);
dis = 0;
for i=1:k
    dis = dis+min(norm(U1(:,i)-U2(:,i),'fro')^2,norm(U1(:,i)+U2(:,i),'fro')^2);
end
dis = sqrt(dis);
end
% tic;
% [U, V, d] = svd_Lineartime(A,k,2*k,opts);
% time(1,1) = toc;
% err_U(1,1) = compare_dist(U,Ut(:,1:k))/Unorm;
% err_V(1,1) = compare_dist(V,Vt(:,1:k))/Vnorm;
% err_sigma(i,1) = norm(d-St(1:k),'fro')/Snorm;
