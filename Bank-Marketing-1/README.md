This task is an explorative Machine Learning task. Part1 is binary classification and part2 is multi-class classification. The Portuguese Bank Marketing dataset "bank-additional-full.csv" is used for this task which is available in this repository. It is also downloadable thrrough "https://archive.ics.uci.edu/ml/datasets/Bank+Marketing" link.

Part1: Binary I first import the data into my machine learning environment. Next, I construct four different ML models using a decision tree, a Naïve Bayesian learner, a support vector machine, and a k‐nearest neighbor classifier. The aim of this binary learning task is to predict whether a client will purchase a product from the bank, i.e. the "output" variable (desired target) is feature 21, with classes 'yes' and 'no'. Below items are investigated:

1- Four models constructed by the algorithms 2- Four confusion matrices corresponding to the models 3- The ROC Curves to contrast the four models.

Part2: Multi-class learning If we suppose that the class label has been changed to feature 15, namely the outcome of the previous marketing campaign, with values 'failure', 'nonexistent' and 'success':

1- A decision tree algorithm is built to construct a model against this multi-class problem. 2- The ROC Curves using one-versus-one comparisons are illustrated. 3- The AUC when using the one-versus-one scheme is calculated (using both the macro average and a prevalence-weighted average)
