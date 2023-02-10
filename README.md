# Credit_Card_Fraud_Detection

## **Problem Statement:** 
* For many banks, retaining high profitable customers is the number one business goal. Banking fraud, however, poses a significant threat to this goal for different banks. In terms of substantial financial losses, trust and credibility, this is a concerning issue to both banks and customers alike.
* In the banking industry, credit card fraud detection using machine learning and deep learning is not only a trend but a necessity for them to put proactive monitoring and fraud prevention mechanisms in place. Machine learning is helping these institutions to reduce time-consuming manual reviews, costly chargebacks and fees as well as denials of legitimate transactions.
* In this project we will detect fraudulent credit card transactions with the help of Machine learning and deep learnig models.
* We will analyse customer-level data that has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group. 


## **DATASET DESCRIPTION**
* The dataset contains 284,807 transactions among which there are 492 i.e., 0.172% transactions are fraudulent transactions
* It also contains transactions made by a cardholder in 2 days in month of september 2013 
* This dataset is highly unbalanced. Due to security reasons, most of the features in the dataset are transformed using principal component analysis (PCA). V1, V2, V3,…, V28 are PCA applied features and rest features include ‘time’, ‘amount’ and ‘class’ are non-PCA applied features

## **Table of Contents**
1. [Importing dependencies](#p1)
2. [Exploratory data analysis](#p2)
3. [Splitting the data into train & test data](#p3)
4. [Model Building](#p4)
   * [Perform cross validation with RepeatedKFold](#p4-1)
   * [Perform cross validation with StratifiedKFold](#p4-2)
   * [RandomOverSampler with StratifiedKFold Cross Validation](#p4-3)
   * [Oversampling with SMOTE Oversampling](#p4-4)
   * [Oversampling with ADASYN Oversampling](#p4-5)
5. [Hyperparameter Tuning](#p5)
6. [Conclusion](#p6)


### Note: Save the ML model in model pkl format and deep learning model in h5 format to use trained model
```
Example:
# Deep Learning Models
from tensorflow.keras.models import save_model
save_model(model, "model.h5")
from tensorflow.keras.models import load_model
# load model
model = load_model('model.h5')

# Machine Learning models
import pickle
pickle.dump(model, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

```
