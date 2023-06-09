In this task, I work on the classification techniques for an unbalanced dataset. The Portuguese Bank Marketing dataset "bank-additional-full.csv" is used for this task which is available in this repository. It is also downloadable through "https://archive.ics.uci.edu/ml/datasets/Bank+Marketing" link.

This project contains **two parts**. The first part is to balance an unbalanced dataset after which I will use different classification algorithms to predict the target value of the new balanced dataset. In part2, I will use three more datasets as benchmarks and run classification algorithms on them as well. It is followed by a Friedman’s test to determine whether there is a significant difference between the achieved accuracies of the classification algorithms.


***Part1***

I rebalance the data set using three different below approaches:

 1- Oversampling of the minority class.
 
 2- Under-sampling of the majority class.
 
 3- Balanced sampling, i.e. combining oversampling and under-sampling.
 
I then apply the four classification algorithms, i.e. support vector machine (SVM), decision tree, Naïve Bayesian learner and the k‐nearest neighbor (k-NN)), to the
three resampled data sets. I also use tenfold cross validation for the better classification.

Next, I construct two extra models using the random forests (RFs) and extreme learning trees algorithms, again using tenfold cross validation.

A table will be created showing the accuracies of the six algorithms against each one of the ten folds when trained using the best sampling technique.

I will next determine whether there is a statistically significant difference in the accuracies obtained by the six algorithms against this dataset.

Next, I apply two different feature selection techniques to the data.

Then, I retrain the “best two” classsification algorithms, with new selected features to determine whether feature selection led to improvements in accuracies. 



***Part2***

Three benchmarking datasets are considered for this part, together with the Portuguese Bank Marketing dataset.
- https://archive.ics.uci.edu/ml/datasets/Labor+Relations
- https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
- https://archive.ics.uci.edu/ml/datasets/Iris

I Apply the SVM, k-NN and RFs algorithms to the three new data sets, using tenfold cross validation, to obtain the average accuracies over the ten folds.

A table showing the average accuracies of the three algorithms against all four data sets is then illustrated.

Next, I use Friedman’s test to determine whether there is a statistically significant difference in the accuracies obtained from SVM, k-NN and RFs. I calculate the critical difference (CD) and draw the Nemenyi diagram.
