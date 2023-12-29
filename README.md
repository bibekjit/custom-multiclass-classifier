# ELUClassifier

sklearn style implementation of a multiclass classifier using numpy. 
Classifier uses ELU activation and converts the logits to multiclass probabilities using softmax

network pipeline ->

`z = wx + b  
z = elu(z)  
z = batch_norm(z) (if batch norm is implemented)  
z = dropout(z)  
p = softmax(z)  
loss = crossentropy(y,p)`

# default hyperparameters ->

At default classifier uses SGD optimizer with a learning rate of 0.01 and 100 epochs.
The lr is constant and no batch normalization, dropout or l2 regularization is applied
the model will stop training if it didn't converge for 3 continous epochs, early stopping



