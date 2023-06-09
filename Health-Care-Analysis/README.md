This project involves the analysis of health care data, as obtainable from "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/".
Specifically, the dataset contains the information about patients with diabetes in 130 hospitals in the USA for the years 1999 to 2008. The study consists of approximately 100,000 patients with 55 features, some containing missing values.

The original study, as published in "https://www.hindawi.com/journals/bmri/2014/781670/", looks into the impact of haemoglobin (HbA1c) measurement, which refers to the average level of blood sugar over the past 2 to 3 months, on hospital re-admission rates. The authors used multivariable logistic regression in their study. In this project, my aim is to assess the value of machine learning algorithms, and notably supervised and semisupervised learning techniques, when applied to this data. Below parts are implemented one by one in Project-Code.ipynb.


***1- Feature engineering***

As a first step, I explore the data to obtain a general understanding of the problem domain. Dealing with Null Values, Dimensionality Reduction, Feature Transformation, Normalization and Feature Selection are all performed for preprocessing of data. This step also involves calculating the levels of class imbalance in the dataset, as I move towards supervised learning.


***2- Predicting if the patient is readmitted ("readmitted" feature prediction)***

Now, I build 6 different classifiers, i.e. SVM, Naibe Bayes, KNN, Decision Tree, Boosting and Baging, to predict the outcome in terms of patient re-admissions. This is a multi-class learning problem, with three class labels {no,readmitted within 30 days, readmitted after 30 days}. Various performance critria, such as overall accuracy, f-measure (trade-off between precision and recall) and the runtime, are calculated to compare different models. Area under curve is also calculated for the models.


***3- Predicting gender class***

As another exploratory task, I explore the data by using gender as class label to provide us with additional insights.
Various algorithm from the different algorithm families are built: trees, linear models such as neural networks and support vector machines, distance-based algorithms, Bayesian approaches and ensembles.


***4- Semi-Supervised learning***

Semi-supervised learning, where we address the scenario where most class labels are unknown, is a very common technique used in real-world applications. In this approach, the aim is to combine a small amount of labeled data with a large amount of unlabeled data during training. Semi-supervised learning is based on the observation that unlabeled data, when used in conjunction with a small amount of labeled data, can produce considerable improvement in learning accuracy.

During this part of the project, I implement three different semisupervised approaches for the hospital re-admissions prediction task. In fact, I test various levels of unlabelled data, notably 0% (fully supervised - baseline), 10%, 20%, 50%, 90% and 95%. The semi-supervised approaches I implement are Label-Propagation, Label-Spreading, and Self-Training mthods. The prformance criteria for the classifiers built with Label-Propagation, Label-Spreading, and Self-Training mthods are then calculated and ROC curves are plotted.
