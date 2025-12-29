function W = construct_W(Kn,N,D,X,neighborhood)
% Compute W by minimizeing the cost function E_LLE(W) = sum(square(xi - sum(Wij*xj)))
if(Kn>D) 
  tol=1e-5; % regularlizer in case constrained fits are ill conditioned
else
  tol=0;
end

W = {} ;
for i = 1:N
    k = size(neighborhood{i}, 1) ;
    if (k>D) 
        z = X(neighborhood{i},:) - repmat(X(i,:),k,1); % shift ith pt to origin  K*D
        G = z*z'; % local covariance  K*K

        if trace(G) == 0
            G = G + eye(k,k)* tol; % regularlization
        else 
            G = G + eye(k,k)* tol * trace(G); % regularlization
        end

        w = G\ones(k,1); % solve Gw = 1
        w = w/sum(w); % normalize

    else
        w = NaN;
    end

    W{i} = w ;
end
