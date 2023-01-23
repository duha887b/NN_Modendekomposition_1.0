function [gradients, loss] = modelGradients(dlnet, dlX, T)

% Forward data through the dlnetwork object.
dlY = forward(dlnet,dlX);

% Compute loss.
loss = mse(dlY,T);
% Compute gradients.
gradients = dlgradient(loss,dlnet.Learnables);

end
