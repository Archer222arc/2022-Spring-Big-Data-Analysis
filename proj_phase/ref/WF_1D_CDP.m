% Implementation of the Wirtinger Flow (WF) algorithm presented in the paper 
% "Phase Retrieval via Wirtinger Flow: Theory and Algorithms" 
% by E. J. Candes, X. Li, and M. Soltanolkotabi

% The input data are coded diffraction patterns about a random complex
% valued 1D signal. 


%% Make signal
n = 128;
x = randn(n,1) + 1i*randn(n,1);

%% Make masks and linear sampling operators

L = 6;                   % Number of masks  

% Sample phases: each symbol in alphabet {1, -1, i , -i} has equal prob. 
Masks = randsrc(n,L,[1i -1i 1 -1]);

% Sample magnitudes and make masks 
temp = rand(size(Masks));
Masks = Masks .* ( (temp <= 0.2)*sqrt(3) + (temp > 0.2)/sqrt(2) );

% Make linear operators; A is forward map and At its scaled adjoint (At(Y)*numel(Y) is the adjoint) 
A = @(I)  fft(conj(Masks) .* repmat(I,[1 L]));  % Input is n x 1 signal, output is n x L array
At = @(Y) mean(Masks .* ifft(Y), 2);            % Input is n x L array, output is n x 1 signal          

% Data 
Y = abs(A(x)).^2;  

%% Initialization

npower_iter = 50;                          % Number of power iterations 
z0 = randn(n,1); z0 = z0/norm(z0,'fro'); % Initial guess 
for tt = 1:npower_iter, 
    z0 = At(Y.*A(z0)); z0 = z0/norm(z0,'fro');
end

normest = sqrt(sum(Y(:))/numel(Y)); % Estimate norm to scale eigenvector  
z = normest * z0;                   % Apply scaling 
Relerrs = norm(x - exp(-1i*angle(trace(x'*z))) * z, 'fro')/norm(x,'fro'); % Initial rel. error

%% Loop

T = 2500;                           % Max number of iterations
tau0 = 330;                         % Time constant for step size
mu = @(t) min(1-exp(-t/tau0), 0.2); % Schedule for step size

for t = 1:T,
    Bz = A(z);
    C  = (abs(Bz).^2-Y) .* Bz;
    grad = At(C);                    % Wirtinger gradient
    z = z - mu(t)/normest^2 * grad;  % Gradient update 
    
    Relerrs = [Relerrs, norm(x - exp(-1i*angle(trace(x'*z))) * z, 'fro')/norm(x,'fro')];  
    max(abs(grad))
end

%% Check results

 fprintf('Relative error after initialization: %f\n', Relerrs(1))
 fprintf('Relative error after %d iterations: %f\n', T, Relerrs(T+1))
 
 figure, semilogy(0:T,Relerrs) 
 xlabel('Iteration'), ylabel('Relative error (log10)'), ...
     title('Relative error vs. iteration count')
 
 
