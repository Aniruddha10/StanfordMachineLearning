function J = costFunction(X,y,theta)
m = size(X,1);
prediction=X*theta;
sqrerrors = (prediction-y).^2;
J = 1/(2*m) * sum(sqrerrors);
endfunction