function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% call sigma
sigmoidOneValue = sigmoid(X*theta);

% call cost function
forOneValue = y.*log(sigmoidOneValue);
forZeroValue = (1-y).*log(1-sigmoidOneValue);
cost=(-1)*(sum((forOneValue+forZeroValue)))/m;

% calculate regularization value
regularizedVectorSqr = (theta(2:length(theta),:)).^2;
regularizationValue = sum(regularizedVectorSqr)*lambda/(2*m);

%cost function
J=cost + regularizationValue;

gradOne = ((X(:,1))' * (sigmoidOneValue - y))/m;
%gradRest = ((X(:,2:size(theta) - 1))' * (sigmoidOneValue - y)) / m + (lambda/m)*theta(2:(size(theta) - 1));
gradRest = ((X(:,2:size(theta)))' * (sigmoidOneValue - y)) / m + (lambda/m)*theta(2:(size(theta)));
grad = [gradOne;gradRest];



% =============================================================

end
