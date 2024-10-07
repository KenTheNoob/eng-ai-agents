Assignment-1b<br>
10/06/2024<br>
All equations are shown in 02-assigment-1.ipynb as markdown cells under the output<br>
The first code cell for Logistic Regression has a "Do not run" comment at the start since the cell preprocesses the data and creates a csv file which does not need to be recreated<br>
Stochastic Gradient Descent(additional details in markdown cell):<br>
* Uses 5 hyperparameters: learning_rate, lambda, mini_batch size, momentum, and learning rate decay
* Uses l2 norm as the loss function
* Implements momentum(weighted average of gradients)
* Has an additional learning rate decay
* Can be run with 10000+ iterations or smaller batch size to reduce runtime at the cost of accuracy
* First graph is the training data and target function
* Second graph are the values of the weights
* Third graph is the loss vs epoch plot
* Final plot compares the final hypothesis to the target function<br>
Logistic Regression:<br>
* Similar to previous part but with cross entropy loss and a different gradient
* Data must be preprocessed
* Uses different hyperparameters
* The precision vs recall curve shows how as recall increases, precision decreases
* A perfect curve is shown in testing.ipynb to prove that the logistic regression model works
Logistic Regression observations:<br>
* My precision peaked at around 0.4 which is better than rando guessing(~0.175) showing that the model does indeed work, but not very well(the data may have little correlation or need better preprocessing)
* It is unlikely, but weights may grow too large causing the log to return infinity(can be avoided by adjusting hyperparameters, specifically lambda)
* If the mini-batch size is too small, then it may not be an accurate representation of the rest of the dataset
* The loss vs epoch plot failed to converge, possibly because the algorithm had trouble finding a good minima
* Momentum and learning rate decay seemed to do either little or make the model worse, so they were kept small
