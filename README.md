# ML_PDs_Prediction

Developing a predictive model that estimates the probability of default on loans using historical loan data.

# Description

The provided dataset contains detailed historical information about borrowers, including various features that may impact the probability of default. 
The goal of this project is to leverage this data and apply Machine Learning techniques to build a predictive model that can accurately predict the probability of a loan defaulting.

## 1. Data Cleaning

 - The data initially consists of 100 000 observations on 20 variables where 19 are the supposed independent variables and 1 being the dependent variable i.e loan status
 - The data undergo some preprocessing process and every manipulation to the variables is explained in this document.

The following steps were taken to before creation of the model as per variable:

a. loan_id

The first varible 'loan_id'is a string variable represents a unique identifiers for one client in the book and was checked for duplicates, none were found.
Checking for duplicates enables uniqueness within loan ids  as thye existence of duplicates would result in a training set that has the same characteristics as the test set thus 
this may result in biased results and in some instances lead to 'overfitting' i.e when the model learns the training data too well, capturing noise and random fluctuations in the data,
rather than the underlying patterns hence performs too well on the training data but fails to generalize to new, unseen data.

b. gender

Gender is a categorical data type consisting 3 categories i.e. male, female, other. 	The gender variable was encoded using 'one-hot encoding' since its non-binary hence 
cannot take binary encoding of 0s and 1s. One-hot encoding preserves the categorical nature and avoids making ordinal assumptions about the relationships between these categories, 
meaning they cannot be ordered or ranked as they are treated as new variables factors instead.
The variable was checked for missing data, constistency in name spelling no errors were found.

c. disbursement dates

Disbursement dates are in the date format and are written in the formart of 'year month day' with no delimiter between them. The date variable was checked for missing values or blanks,
none were found. The disbursement dates ranged from 2020 to 2023 and this variable was determined not to be a much useful variable for predicting default and was dropped as a predictor 
but was rather used to create or derive new variables that proved to be informative for the default prediction model i.e loan_length 
This variable trade-off speaks to the duration or time period in months at which the loan existed in the loan portfolio books i.e the time elapsed between the date the loan was disbursed and the 
date when the PDs were predicted.

   loan_length = PD_prediction_date - disbursement_date

The 'PD_prediction_date' was note stated or given, hence an assumption was made for the creation of this additional variable.

The disbursement years alone was also deemed not useful in PD prediction as they are historical dates hence the machine learning model will not see these dates in future given that 
the trainig set would have been trained on them. Encoding the 4-year period may increase dimensionality in the data, this sparsity can make it more difficult for the model to learn 
meaningful patterns and relationships in the data. 
With high-dimensional data, machine learning models have the potential to overfit the training data, capturing noise and irrelevant
patterns that do not generalize well to new, unseen data.
The more dimensions (features) the model has access to, the more complex it can become, and the more it can 'memorize' the training data instead of learning the underlying patterns.
The years alone would only be suitable for cyclical trend visualisation in relation to default or tied to other 
variables for the concepts of 'Loan Seasoning and Default Risk' and 'Loan Monitoring and Early Repayment'

Loan Seasoning and Default Risk:

 - As a loan matures over time, the risk of default generally increases especially for loans disbursements a while ago and have longly existed in the loan portfolios.
 - Newly disbursed loans have a lower default probability, as borrowers are more likely to make their payments on time during the initial period.
 - However, as the loan ages, the likelihood of the borrower encountering financial difficulties, job loss, or other life events that could lead to default increases.

Loan Monitoring and Early Repayment:

 - Lenders typically have more intensive monitoring and follow-up processes for recently disbursed loans.
 - In the early stages of a loan, lenders can more quickly identify and address any signs of delinquency or repayment issues.
 - As a loan ages, the level of monitoring and proactive intervention by the lender may decrease, allowing potential default risks to escalate.

d. currency

The 'currency' variable is a norminal data type with only a single factor 'USD'.
The currency variable was checked for missing values or blanks for which none were found. 
The currency variable had inconsistent labeling, with some values using 'USD' and others using '$USD'.
Specifically, some values were labeled as 'USD' while others were labeled as '$USD'. 
To address this problem, all the currency values were normalized to use the 'USD' label consistently.

The currency variable was deemed less useful in determining PDs and was dropped as a predictor since the variable contained only a 
single factor as explained below.


Lack of Variability

When a variable contains only a single factor or value (in this case, only USD), it lacks variability or differentiation in the data.
Variables with low or to no variability are often not very useful for predictive modeling, i.e. determining PDs, as they do not provide enough information to 
distinguish between different observations or groups.

Encoding Limitations

For variables with a single factor, the standard encoding techniques, such as one-hot encoding, would result in all observations being represented by a single column of zeros,
as there is no variation to differentiate the observations thus does not allow the algorithms to learn patterns and relationships.
This lack of informative encoding reduces the variable's ability to contribute to the predictive power of the model when determining PDs.


e. country

The country variable is a single factor nominal variable similiar to 'currency' variable consisting of 'Zimbabwe' as a factor response.
Incostistences in factor response labels were detected where some were labeled 'Zimbabwe' and some labelled 'Zim'.
The incosistences were handled by replacing those labelled 'Zim' with the label 'Zimbabwe'.
The variable had 100 blanks/missing values and this problem was solved by using location as a determinant of country. 
Upon vizualization of the 'location' variable,using expert-jugdement and general knowledge all locations were within Zimbabwe hence the blanks were relaced with 'Zimbabwe' as country.

This variable was also tagged as less useful in determining PDs with similar explanations as the 'currency' variable i.e single-factor variable creates noise in the data as they 
can only be encoded to only zeros. Refer to (d) currency. Hence the variable was dropped as a predictor to PDs.

f. sex

The sex variable is a categorical data type with factors male, female and other. 
This variable is useful but was instantly dropped from the data since there already exist the 'gender' variable in the data with the exact same resposes aligning to the 'sex' variable,
hence a repitition/duplication of the same variable creates reduntancy, overfitting and multicollinearity in the data.
Multicollinearity refers to when two or more variables in a model are highly correlated with each other.

g. is_employed

The 'is_employed' variable is a character/categorical data type with response factors i.e. True and False. 
This variable was checked for missing values and labelling consistence, none were found.
The variable was encoded using binary encoding that takes up True as 0 and False as 1.

h. job

This variable consisted of more than 4,000 entries with missing data.
Besides missing data the response labels were not inconsistent e.g 'Software Developer' vs 'SoftwareDeveloper' and 'Data Scintist' vs 'Data Scientist'
These missing values were filled using the 'is_emplyed' variable.
It was found that all the missing values on 'job' were not employed under the 'is_employed' variable i.e they were all 'FALSE'
The 'is_employed' variable determined the replacements on the 'job' variable as follows:

if 'is_employed' is 'FALSE' then job is 'Unemployed'

i. Location

This variable was checked for inconsistences and some were found.
The variable had inconsistent response labelling, were the responses had leading spaces hence the names were repeated e.g 'Gweru', ' Gweru ', ' Gweru '.
The variable also had missing data.

j. loan_amount
Loan amount is an integer data type, representing the initial loan amount at disbursement
It has no missing values and is not a repeated variable.

k. number_of_defaults
This variable is of the integer data type, representing the number of type a lender has defaulted in the period of 2020 - 2023
One inconsistence that was found on this variable is its repeatation hence the repeated variable was dropped


l. Float variables
oustanding balance, interest_rates, salary
These variables were defined as float data types hence they were cleaned together.

Missing values
In this case none were found to have missing values.
This approach calculates the mean of the non-missing values in the float variable.
Replace the missing values with the calculated mean or median.
This approach assumes that the missing values are missing at random and can be reasonably approximated by the central tendency of the observed data.
Replacing missing values with the mean or median helps preserve the overall distribution of the variable, as the imputed values are based on the observed data.
This can be important for maintaining the statistical properties of the variable, such as its central tendencyand spread, which may be important for certain types of analysis or modeling.

m. age
this variable represents the age of the lenders.
One problem found on this variable is its repeatation, hence the repeated variable was dropped.


n. remaining term
Assuming 'remaining term' variable is in months, thus represents the number of months left before maturity date of the loan. -Inconsistences in terms of response values were found, with number values inconsistences e.g '72_', '66_'

o. marital_status
This variable is an object datatype with missing values that needs fixed.
The missing respones were replaced with "unknown".
A number of options were available for this variable i.e. dropping the rows with empty marital_status, dropping the marital status variable and replacing with a dummy.
The other choices were not selected beacuse 'dropping the rows with empty marital_status' would result in loss of over 3,000 worth of data points for our machine learning model.
dropping the marital status variable will result in the loss of an important predictor variable which would limit the number of variables for the model.


p. Loan Status
This an object data type variable was investigated for any consistences , none were found


2: Basic EDA (Exploratory Data Analysis):

Key relationships and findings

Findings

The data had 85% of loans that did not default and 15% that did defaulted.
Married lenders constuted 45% of the data , with the least had unknown marital status.
Manicaland lenders contributed 26% of the data , the highest of all of them.
4% of the lenders were unemployed, with 16% being engineers. 
Overal, 96% were actually employed.
Gender types had almost a uniform distribution with female, male and other taking 32.7%, 32.3% and 35%
On any variable gender, employement, marital status, province, job there are more lenders that did not default than those defaulted 
The remaining term is almost normally distributed between 10 and 100 months.

It has been noted that foe the married sample on those that did not default, their interest rate follows a Symmetrical Distribution
i.e. When Q1 = Q2, it indicates that the distribution of the data is symmetrical around the median. This suggests that the data is evenly distributed on both sides of the central tendency.
No Skewness
i.e. Skewness refers to the asymmetry of a distribution, where the data is concentrated more on one side of the central tendency.

Multi-colinearity checks
Relationships on correlation
There is a high strong positive relationship (72%) between 'salary' and 'remaining term' of the loan/debt.
This might be an indicator of multi-collinearity in between these variables.

Relationship between the salary and loan status
It is noted that those that defaulted earned a lower salary between 0 and 7,000.
The loan_length is evely distributed with the loan status.
 
3. Feature Selection

Categorical data
chi-square (χ²)
This test was chosen because it is well-suited for datasets where the variables are categorical.
The chi-square test is used to determine whether there is a significant association or relationship between the feature (independent variable) and the target variable.
Features with a higher chi-square statistic and a lower p-value are considered more important and are likely to be selected as relevant features for the model.
In this case, the two test were used simultaneously to determine the most important features
Using chi-square:
-Most important features with a high chi-square score were Province,marital status, gender, is_employed and job in their order of importance.

Using p-values:
-Most important features with a low p value were Province,marital status, gender, is_employed and job in their order of importance.

job and is_employed were dropped due to their limited importance in predicting loan status.

continuous variables
Information Gain/Mutual Information

IG can handle the variable's continuous nature by discretizing the feature into a set of bins or intervals.
Remaining term had the least feature importance Information Gain.

4. Hyperparameter Tuning
Hyperparameter tuning varioed from algorithm to algorithm
The method used for Hyperparameter Tuning is 'Grid search'
Grid Search is a hyperparameter optimization technique that exhaustively searches through a specified parameter grid to find the best combination of hyperparameters for your machine learning model.

Grid Search will evaluate all possible combinations of these hyperparameter values and select the best performing set of hyperparameters based on the chosen evaluation metric (e.g., accuracy, F1-score, etc.).

The choice of this approach was based on: 

Simplicity and Interpretability:

Grid Search is a straightforward and easy-to-understand approach, making it a good starting point for hyperparameter optimization.
The results from Grid Search are easy to interpret, as you can clearly see the performance of each combination of hyperparameters.

Exhaustive Exploration:
Grid Search explores all possible combinations of the specified hyperparameter values, which can be beneficial when you don't have a strong prior on the optimal hyperparameter values.
This exhaustive approach can help you discover unexpected or non-intuitive combinations of hyperparameters that perform well.

Reproducibility:
Grid Search is a deterministic approach, meaning that running the same Grid Search code with the same parameters will always yield the same results.
This makes Grid Search a good choice when you want to ensure the reproducibility of your hyperparameter tuning process.

Compatibility with Various Algorithms:
Grid Search can be used with a wide range of machine learning algorithms, making it a versatile and widely-applicable hyperparameter optimization technique.
Visualization and Analysis:
The results of Grid Search can be easily visualized and analyzed, such as through the use of heatmaps or parallel coordinate plots.
This can provide valuable insights into the sensitivity of the model's performance to different hyperparameter values.
However, there are also some limitations to Grid Search:

5. Cross Validation:

K-Fold Cross-Validation
The K Fold Cross validation used using the cross_val_score function from the sklearn.model_selection module

The choice of the cross validation was determined by:

Robust Performance Estimation:
K-Fold Cross-Validation provides a more reliable estimate of the model's performance compared to a single train-test split. By using multiple folds, it helps to mitigate the impact of a particular train-test split being biased or not representative of the overall data distribution.

Efficient Use of Data: 
K-Fold Cross-Validation allows you to use all of your available data for both training and evaluation. In each fold, a portion of the data is used for training, and the remaining portion is used for evaluation. This is more efficient than a single train-test split, where a portion of the data is left unused.

Hyperparameter Tuning: 
When tuning hyperparameters, K-Fold Cross-Validation can provide a more stable and reliable evaluation of the model's performance for different hyperparameter settings. This helps you to select the best set of hyperparameters more confidently.

Generalization Ability: 
By evaluating the model's performance on multiple held-out folds, K-Fold Cross-Validation can give you a better sense of how the model will generalize to new, unseen data.


Flexibility: 
K-Fold Cross-Validation is a flexible technique that can be applied to a wide range of machine learning problems and models. It can be used for classification, regression, and other types of tasks.

6.Feature Scaling and Transformation:

Normalization was chosen as the Feature Scaling and Transformation method because of varios reasons that includes:
Normalization, also known as min-max scaling or feature scaling, transforms the feature values to a common range, typically between 0 and 1.
This is particularly useful when the features have vastly different numerical ranges, as it can help ensure that no single feature dominates the others during model training.

a. Data Visualization:
Normalized features can be easier to visualize, as they all have a common scale, making it easier to compare and interpret the data.

b. Interpretability:
Normalized features are often easier to interpret, as the values represent the relative importance of each feature within the range of 0 to 1.
This can be particularly useful for feature importance analysis or when communicating model results to stakeholders.

c. Improved Algorithm Performance:
Many machine learning algorithms, such as linear regression, logistic regression, and neural networks, are sensitive to the scale of the input features.
Normalization can help improve the performance of these algorithms by ensuring that all features contribute equally to the model.

d. Numerical Stability:
Normalization can help prevent numerical instability or overflow/underflow issues in certain algorithms, especially when working with large or small feature values.
Algorithms like gradient descent can be more stable and converge faster when the features are on a similar scale.



Evaluation Cross validation results

i. Logistic Regression

The mean cross-validation accuracy of the Logistic Regression model is 0.88.
This indicates that the Logistic Regression model achieves an accuracy of 88% on average when evaluated using cross-validation.

ii. Random Forest Classifier

The mean cross-validation accuracy of the Random Forest Classifier is 0.89.
This suggests that the Random Forest Classifier performs slightly better than the Logistic Regression model, with an average accuracy of 89%.

iii. Decision Tree Classifier

The mean cross-validation accuracy of the Decision Tree Classifier is 0.88.
The Decision Tree Classifier performs similarly to the Logistic Regression model, with an average accuracy of 88%.

iv. Gradient Boosting Classifier

The mean cross-validation accuracy of the Gradient Boosting Classifier is 0.90.
This is the highest accuracy among the models presented, indicating that the Gradient Boosting Classifier is the best-performing model for your classification problem, with an average accuracy of 90%.

v.Linear Discriminant Analysis (LDA)

The mean cross-validation accuracy of the LDA model is 0.87, with a standard deviation of 0.01.
LDA has the lowest average accuracy among the models presented, with an accuracy of 87%.

7. Model Building
 
Models trained includes:

Logistic Regression (LR)
Random Forest Classifier (RFC)
Desicion Tree Classifier (DTC)
Gradient Boosting Classifier (GBC)
Linear Discriminant Analysis (LDA)

a. Logistic Regression (LR)
How it works:

Inputs and Outputs: 
Logistic Regression takes a set of input features (X) and predicts the probability of a binary or categorical output (y). The output is typically a value between 0 and 1, which can be interpreted as the probability of the instance belonging to a particular class.

Logistic Function: 
Logistic Regression uses the logistic function, which is an S-shaped curve, to map the input features to the output probability. The logistic function is defined as: p(x) = 1 / (1 + e^(-z)) where p(x) is the predicted probability, and z is a linear combination of the input features and their corresponding weights.

Model Training: 
During the training process, Logistic Regression learns the optimal weights (coefficients) for the input features, which determine how much each feature contributes to the final prediction. The algorithm tries to find the weights that minimize the error between the predicted probabilities and the true class labels.

Classification Threshold: 
To make a final classification decision, a threshold is typically set (often at 0.5). If the predicted probability is greater than or equal to the threshold, the instance is classified as belonging to the positive class; otherwise, it is classified as belonging to the negative class.

For example in a loan application problem. The goal is to predict whether a loan applicant will default on their loan (binary classification).

lets say the input features are:

Credit score (numerical)
Income (numerical)
Employment status (categorical: employed, unemployed)
Loan amount (numerical)
The output (target variable) would be a binary value: 0 (won't default) or 1 (will default).

During the training process, Logistic Regression would learn the weights (coefficients) for each input feature, such that the model can accurately predict the probability of an applicant defaulting on the loan. For example, it might learn that a higher credit score and income decrease the probability of default, while a higher loan amount increases the probability of default.

Once the model is trained, it can be used to predict the probability of default for new loan applicants. If the predicted probability is greater than or equal to 0.5, the applicant would be classified as likely to default on the loan.

Reason of choice:
Logistic Regression is a powerful and widely-used algorithm for binary and multi-class classification problems in machine learning, due to its simplicity, interpretability, and effectiveness in many real-world applications.


b. Decision Trees
 
How it works:
Decision Trees are another popular machine learning algorithm used for both classification and regression problems. In the case of classification problems, Decision Trees work as follows:

Inputs and Outputs:
Decision Trees take a set of input features (X) and predict the class label (y) for each instance.

Tree Structure: 
Decision Trees are built with a tree-like structure, consisting of nodes, branches, and leaves. The internal nodes represent the features used for making a decision, the branches represent the decision rules, and the leaf nodes represent the final class predictions.

Building the Tree: 
The algorithm starts by selecting the most informative feature to split the data at the root node. This is typically done by evaluating a metric like Information Gain or Gini Impurity, which measures the "purity" or "homogeneity" of the data at each node.

Recursive Splitting: 
The algorithm then recursively splits the data based on the selected feature, creating child nodes. This process continues until a stopping criterion is met, such as a maximum depth of the tree or a minimum number of samples in a node.

Leaf Node Predictions: 
At the leaf nodes, the algorithm assigns the most common class label among the samples that reach that node.

Classification: 
To classify a new instance, the algorithm starts at the root node, follows the appropriate branches based on the instance's feature values, and ends up at a leaf node, which provides the predicted class label.

Example:

Let's consider a simple example of a Decision Tree for predicting whether a person will buy a product or not (binary classification).

The input features might be:

Age (numerical)
Income (numerical)
Education level (categorical: high school, college, graduate)
Marital status (categorical: single, married, divorced)
The output (target variable) would be a binary value: 0 (won't buy) or 1 (will buy).

The Decision Tree algorithm might start by selecting the "Income" feature as the root node, as it has the highest Information Gain or lowest Gini Impurity. It might then split the data based on the Income feature, creating two child nodes: one for "Income <= $50,000" and another for "Income > $50,000".

The algorithm would then continue to recursively split the data based on other features, such as Age or Education level, until it reaches the leaf nodes. At the leaf nodes, the algorithm would assign the most common class label (e.g., "will buy" or "won't buy") based on the samples that reach that node.

To classify a new instance, the algorithm would start at the root node, follow the appropriate branches based on the instance's feature values, and end up at a leaf node, which would provide the predicted class label (e.g., whether the person will buy the product or not).

Reason of choice:

Decision Trees can capture complex non-linear relationships in the data and handle both numerical and categorical features

c. Random forest

Random forest works as follows:

Multiple Decision Trees: 
Random Forest creates a "forest" of multiple Decision Trees, each trained on a different subset of the training data.

Bagging (Bootstrap Aggregating): 
To create each Decision Tree, Random Forest uses a technique called Bagging. It randomly selects a subset of the training data with replacement (known as a bootstrap sample) and trains a Decision Tree on that subset.

Feature Randomness: 
In addition to Bagging, Random Forest also adds another layer of randomness by randomly selecting a subset of features to consider at each node split in the Decision Trees. This helps to decorrelate the trees and improve the overall model performance.

Ensemble Prediction: 
To make a prediction on a new instance, Random Forest collects the predictions from all the individual Decision Trees in the forest. The final prediction is made by either taking the majority vote (for classification) or averaging the predictions (for regression).

Example:

Let's consider the same example as before, where we want to predict whether a person will buy a product or not (binary classification).

The input features are:

Age (numerical)
Income (numerical)
Education level (categorical: high school, college, graduate)
Marital status (categorical: single, married, divorced)
The output (target variable) is a binary value: 0 (won't buy) or 1 (will buy).

In this case, the Random Forest algorithm would:

Create multiple Decision Trees (e.g., 100 trees) by randomly selecting subsets of the training data and features for each tree.
Train each Decision Tree independently on its respective subset of the data.
To classify a new instance, the algorithm would pass the instance through each of the 100 Decision Trees in the forest and collect the individual predictions.
The final prediction would be made by taking the majority vote of the 100 individual predictions. If 60 out of the 100 trees predict "will buy", the final prediction would be 1 (will buy).

Reason for choice:

The key advantages of Random Forest are:

Improved Accuracy: 
By combining multiple Decision Trees, Random Forest can often achieve higher classification accuracy than a single Decision Tree.

Robustness to Overfitting: 
The randomness introduced in the Bagging and feature selection steps helps to prevent the individual Decision Trees from overfitting the training data.

Handling of Diverse Data Types: 
Random Forest can handle both numerical and categorical features, making it a versatile algorithm for a wide range of classification problems.

Feature Importance: 
Random Forest can provide insights into the relative importance of each input feature, which can be useful for feature selection and understanding the underlying patterns in the data.

d. Gradient Boosting

How it works:

Weak Learners: 
Gradient Boosting uses a series of "weak learners", which are typically simple models like Decision Trees with a limited depth (e.g., Decision Stumps with only one split).

Sequential Training: Unlike Random Forest, which trains each Decision Tree independently, Gradient Boosting trains the models sequentially. Each new model is trained to predict the residual errors (the difference between the true labels and the current model's predictions) of the previous model.
Iterative Improvement: 
The algorithm starts with a simple initial model (e.g., a model that predicts the overall mean or median of the target variable). It then iteratively adds new weak learners to the ensemble, with each new model focusing on correcting the mistakes made by the previous models.

Gradient Descent Optimization: 
Gradient Boosting uses a technique called Gradient Descent to optimize the parameters of the weak learners. The algorithm calculates the gradient (the direction and magnitude of the error) and updates the model parameters to minimize the overall error.

Ensemble Prediction: 
To make a prediction on a new instance, Gradient Boosting combines the predictions of all the weak learners in the ensemble, typically by summing their weighted predictions.

Example:

Let's consider the same example as before, where we want to predict whether a person will buy a product or not (binary classification).

The input features are:

Age (numerical)
Income (numerical)
Education level (categorical: high school, college, graduate)
Marital status (categorical: single, married, divorced)
The output (target variable) is a binary value: 0 (won't buy) or 1 (will buy).

In this case, the Gradient Boosting algorithm would do the following:

Start with a simple initial model, such as a Decision Stump that predicts the overall proportion of people who will buy the product.
Train the first weak learner (a shallow Decision Tree) to predict the residual errors between the true labels and the initial model's predictions.
Add the first weak learner to the ensemble and update the model's predictions accordingly.
Train the second weak learner to predict the new residual errors, and add it to the ensemble.
Repeat step 4, adding new weak learners to the ensemble iteratively, with each new model focusing on correcting the mistakes made by the previous models.
To classify a new instance, the algorithm would combine the predictions of all the weak learners in the ensemble, typically by summing their weighted predictions.
The key advantages of Gradient Boosting are:

Reason for choice:

Powerful Performance: 
Gradient Boosting can often achieve state-of-the-art performance on a wide range of classification (and regression) problems.

Automatic Feature Selection: 
Gradient Boosting can automatically determine the importance of each input feature and focus on the most informative ones.

Handling of Diverse Data Types: 
Like Random Forest, Gradient Boosting can handle both numerical and categorical features.

Interpretability: 
The individual weak learners (Decision Trees) in the Gradient Boosting ensemble can provide some level of interpretability, though the overall model can be more complex to interpret compared to a single Decision Tree.

e. Linear Discriminant Analysis

How it works:

Assumptions: 
LDA assumes that the data for each class follows a multivariate normal distribution, and that the classes have the same covariance matrix.

Class Means and Covariance: 
LDA first calculates the mean vector and covariance matrix for each class in the training data.

Discriminant Function: 
LDA then constructs a linear discriminant function that maximizes the separation between the classes. This function is a linear combination of the input features, and it can be used to classify new instances.

Classification: 
To classify a new instance, LDA calculates the discriminant function values for each class and assigns the instance to the class with the highest discriminant function value.

Example:

Let's consider a simple binary classification problem where we want to predict whether a person will buy a product or not (0 for won't buy, 1 for will buy). The input features are:

Age (numerical)
Income (numerical)
In this case, LDA would work as follows:

Assumptions: 
LDA assumes that the ages and incomes of both the "will buy" and "won't buy" classes follow a bivariate normal distribution, and that the covariance matrices for both classes are the same.

Class Means and Covariance: 
LDA calculates the mean age, mean income, and the common covariance matrix for both the "will buy" and "won't buy" classes.

Discriminant Function: 
LDA then constructs a linear discriminant function that takes the age and income of a person as input and outputs a value that represents the likelihood of that person buying the product. This function is a linear combination of age and income, and it is designed to maximize the separation between the "will buy" and "won't buy" classes.

Classification: 
To classify a new person, LDA calculates the discriminant function value for that person. If the value is above a certain threshold, the person is classified as "will buy"; otherwise, they are classified as "won't buy".

Reason for choice:
Simplicity:
LDA is a relatively simple and interpretable algorithm, making it easy to understand and explain.

Efficient: 
LDA is computationally efficient, especially compared to more complex machine learning algorithms.

Robustness: 
LDA is relatively robust to violations of its underlying assumptions, such as non-normal distributions or unequal covariance matrices.


8. Model Evaluation

AUC (Area Under the Curve):

AUC is a performance metric that ranges from 0 to 1, with 1 representing the best possible performance and 0.5 indicating a random classifier.
The AUC values in the table represent the area under the Receiver Operating Characteristic (ROC) curve for each model.
The Gradient Boosting Classifier has the highest AUC of 0.707481, which indicates that it has the best overall classification performance among the models.
The Decision Tree Classifier has the second-highest AUC of 0.643532, followed by the Random Forest Classifier (0.640666) and Logistic Regression (0.622825).
The LDA model has the lowest AUC of 0.633791, suggesting it has the weakest overall classification performance.

Accuracy:

Accuracy is the proportion of correct predictions made by the model out of the total number of predictions.
The Gradient Boosting Classifier has the highest accuracy of 0.90090, meaning it correctly predicts the target class 90.09% of the time.
The Random Forest Classifier has the second-highest accuracy at 0.88825, followed by the Decision Tree Classifier (0.88545), Logistic Regression (0.88025), and LDA (0.88145).
In summary, the Gradient Boosting Classifier stands out as the best-performing model, with the highest AUC and accuracy among the models compared. The Decision Tree Classifier and Random Forest Classifier also perform well, with similar accuracy and AUC values. Logistic Regression and LDA have the lowest performance metrics, indicating they are the weakest models for this particular classification problem.


9. Endpoint Development for Inference:

How to Run the file using the Endpoints
a. Open Anaconda Prompt
b. Choose directory where the project is located using the "cd" command specifying the folder path as follows:

cd C:\Users\ntumbare\Desktop\ML_PDs_Prediction\ML_PDs_Prediction

c. Once the path is defined, type install "fastAPI" and "Uvicorn" packages as follows:

pip install fastapi uvicorn

#The 'uvicorn app:app --reload' command is used to run a FastAPI application using the Uvicorn server. The app:app part refers to the app variable in the app.py file, which is the FastAPI application instance.

The --reload option tells Uvicorn to automatically reload the server when changes are made to the code. 

d. After you have lauched the project copy the hyperlink to the server and past it to any web browser of your choice as follows:

 http:127.0.0.1:800/docs

e. After the Application has been lauched,one can now use it to predict probabilities by entering information.

To predict using the Application
Click on Try it out
f. On Loan ID enter a unique identifier such as ID (a string variable).
g.




10. Data Drift Detection

Data Drift and Model Drift Detection

Data Drift
If there is changes in the data, we normally call it as Data Dift or Data Shift. A data drift can also refer to 
.changes in the input data
.changes in the values of the features used to define or predict a target label.
.changes in the properties of the independent variables.

Model Drift
This refers to changes in the perfomrmance of the model over time. It is the deterioration of models over time in the case of accuarcy and predictin. ML Models do not live in a static environment hence they will deteriorate or decay over time.
Assume you have a simple model 

y = mx + b

y = target  label
x = independent features or predictors

Any changes that takes place in x is termed as data dift i.e feature drift or covariate shift.
Features drift - changes in the features.
Covariate shift - the distribution of the input features (or covariates) used to train a machine learning model is different from the distribution of the input features used when the model is deployed or used in production.

Why monitoring data drift is crucial for model maintenance

Concept Drift:

Machine learning models are trained on historical data, which may not fully represent the current state of the problem.
Over time, the underlying relationship between the input features and the target variable can change, a phenomenon known as concept drift.
If a model is not updated to account for these changes, its performance will degrade, leading to decreased accuracy and reliability.

Data Distribution Shift:
The distribution of the input features used to train the model may differ from the distribution of the input features encountered in production.
This shift in data distribution, known as covariate shift, can cause the model to perform poorly on new, unseen data.
Monitoring data drift helps identify these distribution shifts and triggers the need for model retraining or fine-tuning.

Model Degradation:
As time passes, the performance of a machine learning model can degrade due to the accumulation of small changes in the data.
This gradual degradation can lead to the model becoming increasingly inaccurate and unreliable over time.
Monitoring data drift allows you to detect these gradual changes and take proactive measures to maintain the model's performance.
Regulatory Compliance:

In certain industries, such as finance and healthcare, models used for decision-making must adhere to regulatory requirements.
Monitoring data drift helps ensure that the model's performance remains within acceptable limits, which is crucial for maintaining compliance.

Business Relevance:
Machine learning models are often deployed to solve business problems, and their performance directly impacts the organization's bottom line.
Monitoring data drift allows you to identify when a model's performance is degrading and take appropriate actions to maintain its relevance and effectiveness.

11. Model Analysis:


Model coefficients or feature importances.
a. Logistic Regression

Numerical Features:

salary: 
This feature has the highest negative importance (-10.210559), indicating that as the salary increases, the model is less likely to predict a positive outcome (e.g., loan default).

interest_rate: 
This feature has the second-highest negative importance (-5.816207), suggesting that as the interest rate increases, the model is less likely to predict a positive outcome.

age: 
This feature has a negative importance of -3.412853, meaning that as the age increases, the model is less likely to predict a positive outcome.

outstanding_balance: 
This feature has a relatively low negative importance (-0.118706), indicating that it has a small influence on the model's predictions.

remaining term: 
This feature has a positive importance of 4.629233, suggesting that as the remaining term of the loan increases, the model is more likely to predict a positive outcome.

loan_amount: 
This feature has a positive importance of 3.010017, indicating that as the loan amount increases, the model is more likely to predict a positive outcome.

number_of_defaults: 
This feature has a positive importance of 1.084348, meaning that as the number of defaults increases, the model is more likely to predict a positive outcome.

Categorical Features:

gender: 
The feature importance for gender shows that the model is more likely to predict a positive outcome for "male" (0.810864) and "other" (0.812454) genders, compared to "female" (0.550755).

marital_status: 
The model is more likely to predict a positive outcome for "divorced" (0.887246), "married" (0.632205), and "unknown" (0.625919) marital statuses, compared to "single" (-0.028701).

province: 
The feature importance for different provinces varies, with "Matabeleland North" (3.524731) and "Not_Specified" (3.246138) having the highest positive importance, and "Mashonaland East" (-1.349789) and "Bulawayo" (-1.071605) having the highest negative importance.

-For all the models the highest value represent the importance value of the feature

 Instances where the model performs poorly


model's prediction errors
For logistic regression , the model is perfomming poorly on determining True positives , it has a number of False positives where a loan is said to be a deafult whilst it is a non-defult.
The same applies for Random forest and Gradient Boost.
For Decision tree and Linear Discriminant, the model is failling to identify   true negatives i.e those that defaulted but its taking them as non-default.

Limitations of the Model:

Feature Importance Ranking Sensitivity: 

The feature importance rankings provided are specific to the particular model trained. Different modeling techniques or hyperparameter settings could result in somewhat different importance rankings for the features. The rankings should be interpreted as relative indicators rather than absolute measures.

Potential Overfitting:
With a large number of features relative to the sample size, there is a risk of the model overfitting the training data. This could result in the model performing well on the training set but generalizing poorly to new, unseen data. Further evaluation on held-out test data would be needed to assess generalization.

Extrapolation Challenges: 
The model may struggle to make accurate predictions for instances that fall outside the range of the training data. For example, predicting outcomes for individuals with exceptionally high loan amounts or interest rates that were not well represented in the original dataset.
Demographic Biases: The feature importance rankings suggest the model places significant weight on demographic characteristics like marital status and gender. This raises concerns about potential demographic biases in the model's predictions that would need to be carefully evaluated.

Situations Where the Model May Not Perform Well

Rare or Unusual Cases: Instances with feature combinations that are infrequent in the training data may be more difficult for the model to predict accurately. The model may lack sufficient examples to learn the appropriate patterns for these edge cases.

potential enhancements or future directions for model improvement.

Monitoring and Retraining:
Establish a process for continuously monitoring the model's performance in production, particularly for high-stakes applications.
Implement triggers for model retraining when the underlying data distribution or relationships change significantly over time.
Develop robust feedback loops to incorporate user feedback and real-world performance data into future model iterations.








