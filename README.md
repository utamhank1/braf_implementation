# braf_implementation
Implementation of the Biased Random Forest Machine Learning Algorithm in dealing with class imbalance on the Pima Dataset.

# Requirements:
1. The following python 3.7 modules are used in the execution of this file, non-standard library packages are bolded and version number is given:

python 3.7
sys
os
collections
math
random (1.1)
matplotlib (3.1.3)
concurrent 
arpgarse
__pandas (1.0.2)
numpy (1.18.1)
scipy (1.4.1)
seaborn (0.10)__

The requirements.txt file contains the non-standard library python modules that need to be installed into the virtual environment.

2. One needs to have the Pima diabetes dataset in the same directory that the main file is present.


# Launching the Program:

Enter the following into the command line:

python main.py -f diabetes.csv -K 10 -s 100 -p .7

If you desire to run the program with the optional arguments, the format is as followed:

python main.py -f diabetes.csv -K 10 -s 100 -p .7 -imp random -stdev 3.5 -exp True

## Input: 
There are two seven-supplied inputs to the program doc_parser.py:
1.	-f [file]: This string is the dataset that you wish to run the model on, in this case, should always be diabetes.csv which should be present in the same directory as main().

2.	-K [number of folds]: This is an integer number that specifies the number of folds you would like to divide your training data into for the purposes on K-fold cross validation.
3.	-s [forest size] – This integer is the parameter that sets the number of decision trees to generate while creating the random forest.

4.	-p [proportion of critically sampled data] – This float parameter tells that program what proportion of the training data to sample from the critical dataset for purposes of training the model in the braf algorithm.

5.	-imp [(Optional) imputation_method] – This string parameter tells that program the method you wish to use to impute values for missing data in the diabetes.csv dataset, the default is ‘random’ where a random number is assigned to the missing values in the dataset that are sampled from the gaussian distribution generated from the mean and std. deviation of the column that missing data is located in. Other valid arguments are ‘mean’, where the mean value of that column is imputed to all the missing values, and ‘median’, where the median value of that column is imputed onto the missing values.

6.	-stdev [(Optional) standard deviation to keep] – This float parameter tells the program how many standard deviations of each feature to keep in dealing with removal of outliers. Default is 3.5. 

7.	-exp [(Optional) explore data] – This boolean value tells the program whether the user desires to view correlational charts and histograms related to the distributions and correlations of the features. Default is set to False.

## Output:
0.  There will be a list of precision and recall values from every run of the K-fold cross validation, precision and recall values from the test data, AUROC, AUPRC values printed to the console, and either seven .png files generated (if -exp is set to True), or four .png files generated (if -exp is not passed or set to False) in the working directory with the following titles.
1.	feature_correlations.png – (Generated if -exp set to True) This file displays a correlational heatmap of every feature and the degree of correlation with every other feature.

2.	Outcome_1_histograms.png – (Generated if -exp set to True) This file displays the distributions of all features for which the outcome = 1 in the dataset.

3.	Outcome_0_histograms.png – (Generated if -exp set to True) This file displays the distributions of all features for which the outcome = 0 in the dataset.

4.	PRC_Training Data.png – This file displays the PRC curve, along with the AUPRC for the training data across all K-Folds of the K-Fold cross validation.

5.	ROC_Training Data.png – This file displays the ROC curve, along with the AUROC for the training data across all K-folds of the K-fold cross validation.

6.	PRC_Testing Data.png - This file displays the PRC curve, along with the AUPRC for the testing data.

7.	ROC_Testing Data.png – This file displays the ROC curve, along with the AUROC for the testing data.
