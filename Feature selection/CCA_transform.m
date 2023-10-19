function [P] = CCA_transform(X, Y, parameter)
% Input: 
%       X -- centered training instance matrix (N *d) 
%       Y -- centered training label matrix (N*q)
%       parameter.ratio -- reduced ratio
%       parameter.regX -- regularization constant for X
%       parameter.regY -- regularization constant for Y
% Output: 
%       P -- projection matrix (d*dim) 
% CCA: 
%       [X'*Y*inv(Y'*Y+regY*I)*Y'*X]*P = lambda*(X'*X+regX*I)*P
%-----------------------------------------------------------------
disp('CCA...................................................');
disp(strcat('Regularized constant for X =',num2str(parameter.regX))); 
disp(strcat('Regularized constant for Y =',num2str(parameter.regY))); 
[N1, d] = size(X);
[N2, q] = size(Y);
if(N1 ~= N2) disp('The number of training instances in X and Y is not equal'); end
N =min(N1, N2);

YY=Y'*Y; %q*q
XY=X'*Y; %d*q
XX=X'*X; %d*d

disp(strcat('XX: ', num2str(max(max(XX))), ' to ', num2str(min(min(XX)))));
disp(strcat('YY: ', num2str(max(max(YY))), ' to ', num2str(min(min(YY)))));


%norm(YY-YY','fro')
%norm(XX-XX','fro')

rankYY = rank(YY);
disp(strcat('The ranks of YY = ',num2str(rankYY)));

regY = parameter.regY;
if (regY == 0)
    A = XY*pinv(YY)*XY';
else
    if (regY <0)
        maxYY = max(max(YY)); 
        A = XY*inv(YY - regY*maxYY*eye(q,q))*XY';
    else % regY >=0
        A = XY*inv(YY + regY*eye(q,q))*XY';
    end  
end

A = (A+A')/2.0;

norm_fro= norm(A-A','fro');
if(norm_fro ~= 0)
    disp(strcat('Warning: not real symmetrical matrx = ', num2str(norm_fro)));
end

regX = parameter.regX;
if (regX < 0)
    maxXX = max(max(XX));
    XX = XX - regX*maxXX*eye(d,d);
else
    XX = XX + regX*eye(d,d);
end
    
XX=(XX+XX')/2;

if(parameter.rank==1)
    rankA = rank(A);
    rankXX = rank(XX);
    disp(strcat('The ranks of two matrixes = ',num2str(rankA), ', ',  num2str(rankXX)));
    min_rank = min(rankA, rankXX);
else
    min_rank = q-1;
end

[V, D] = eig(A, XX);

eigenVectors = V;
eigenValues= diag(D);

[eigenValues, eigenVectors] = sort_eigenvalue_descend(eigenValues, eigenVectors);
    
reduced_dimension = detect_reduced_dimension(eigenValues, min_rank, q, parameter.ratio);
    
P = eigenVectors(:, 1:reduced_dimension);

end
