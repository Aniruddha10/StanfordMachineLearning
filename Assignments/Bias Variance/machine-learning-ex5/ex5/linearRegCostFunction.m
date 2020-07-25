function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%iteration = 50;
%J_history = zeros(iteration,1);
%initializeTheta = [1;1];
%alpha = 2;
%for i=1:1
%  mse = sum(((X * initializeTheta) - y) .^ 2);
%  J = (1/(2*m)) * mse;
%  initializeTheta = initializeTheta - ((alpha/m) * ((X * initializeTheta) - y)' * X)';
%  J_history(i) = J;
%endfor
%J_history


mse = (1/(2*m)) * (sum(((X * theta) - y) .^ 2));
rv = (lambda / (2*m)) * (sum((theta(2:end,1) .^ 2)));
J =  mse + rv;


gradOne = (1/m) * ((X (:,1))' * (X * theta - y));  
gradRest = (1/m) * ((X(:,2:end)' * (X * theta - y))) + (lambda / m) * theta(2:end);
grad = [gradOne; gradRest];

% =========================================================================

grad = grad(:);

end
