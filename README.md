# ML_PDs_Prediction

Developing a predictive model that estimates the probability of default on loans using historical loan data.

# Description

The provided dataset contains detailed historical information about borrowers, including various features that may impact the probability of default. 
The goal of this project is to leverage this data and apply Machine Learning techniques to build a predictive model that can accurately predict the probability of a loan defaulting.

## Data Cleaning
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

