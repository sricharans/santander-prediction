## Definition and Introduction

Santander is always looking for ways to help our customers understand their financial health and identify which products and services might help them achieve their monetary goals. 

We come in to identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted.

## Data Source & Overview
We are provided with an anonymized dataset containing numeric feature variables, the binary ‘target’ column, and a string ‘ID_code’ column.

The training dataset consists of 200 numeric variables labelled from var_0 to var_199 which will be used to train the model

## Task
Predict the value of the ‘target’ column in the test set, which is the probability that the user is making a purchase or not.

# Technical Specifications

![img](https://lh3.googleusercontent.com/8C2iNNFOCOnUkV9A7ipCKaj8WOcmOWOOSFHt4trGEeEAz0_v6x-5EZ-C0YDdMjfI1AY29KsuHa4s32y0TdJwstSZlWXDHMvFpdcWs1cLDrY_0OAJYxdMo43ZI6IvZ_Omwo1I6Zxe)
# Exploratory Data Analysis
To understand the raw data better, we had to analyze distributions to check for any inherent patterns. Since our dataset had 200 numeric variables, we chose a couple of variables to identify their distribution
## Class Imbalance
Right off the bat, we notice that we have highly skewed data concerning the target label. There were ~9x more data points for people who did not purchase than for people who did.

![img](https://lh5.googleusercontent.com/8cPWXHOYMKtOu0ckRHK-RdXANHHs4Jwfjn5x8SsauRPDGTBKCKQq63KHbL_ERxNXb6Ut2WF4x-6T4PybMy7hzpDs04bbaKawzc6hqDJ5zlKd8TzQOka6jBaYbkS-FsIIrILD531w)
## Correlation among variables
We plotted a correlation matrix and found that the correlations among the features are very  low.  The below heatmap shows the top 5 most correlated features. 

![img](https://lh3.googleusercontent.com/ttk_cjEEDFv4apWBAXzv608iVKEntvjmQ8bEL4XUPng6LosGg5u21lqPNz9LC3NTcO-GWK5EDBGu_Su8ahIAcmrVzvak1RCbW-ZCc0pFrOByjvzlWJKKnOlKtmxkW_gvPPM9Awhn)

## Synthetic Data
When the number of unique values for each train and test dataset (split by positive and negative labels) was plotted, this was the output:

![img](https://lh5.googleusercontent.com/v8X8GD9flMEQmXiaNKHRoxXxySFJUB0dqzEWwr1aHf3B0VeDlQa-xdc-fxLvfzB-a8ZEtaUke4Cp6dpNG66b3xbsPRO5XNYgRW-_VK7jRC_nsUBjTkqqhMf9EvmrBdKJYt7tfAGw)

There is a significant difference in the unique value numbers of each variable, between training data and test data.

Some noises in the original training data must’ve been added by Kaggle hosts. So if the variable value is unique, this variable may be synthetic (added noise from the original data), so we need to tell this information to the model.

Now we know that noise and synthetic data exist in both training and testing datasets. To tackle this problem, we added 600 new features which included statistics of the pre-existing 200 columns of data, to capture the authenticity of the data point. Due to the high volume of data, some models use a sample of the true data (~116k records)

## Data Scaling
Each feature variable seems to have numeric values from different ranges. To make sure no variable is overwhelmingly affecting the model, we need to transform each feature so that its values are brought to a standard comparable range.

Due to the existence of outliers, the min-max scaler would be ineffective. The outliers would pull the extreme values making the ‘normal’ values obsolete. Hence, we decided to use StandardScaler.
# Model Training
Individual models were trained using the transformed data. To find the best parameters for the models, we used nested K-Fold cross-validation and BayesSearchCV. Once the best parameters were computed, these models were stacked to create an ensemble model.
## Individual Models
### Logistic Regression
Logistic Regression is a form of classification technique used to perform binary or multi-class classification based on a set of independent variables. In binary classification, it uses probability thresholds to predict the class as 0 or 1. To balance the imbalanced classes in our data, we passed ‘class_weights’  as balanced while training the model.  

We got the following best hyperparameters after running GridSearchCV:

```
C: 0.1,
class_weight': 'balanced',  
penalty: 'l1', 
solver: 'saga'
```

![img](https://lh6.googleusercontent.com/5Am27ujp0SC13rsSbLivZh0wtYeeXduzX89hwQD_U4OEU6lgWqG7TaiGx5Hf7sv3MN7nASfHDlHM4qGDT-62wOtVGROx8E_V9KQ_4X3o4WpXa-AybE4vHW_kVFYueHbW0olQGJCr)
![img](https://lh5.googleusercontent.com/wiKqo4Nu1n4x2IKBrw_Vvy0T_8IQR8X71QE4YtdzXbmtjjwlN8xBwveiOpiQPYOzs7u-Lw1U-xIWQkCyRiFIpzk1n_ZEPtofxHH1PEVOXrKXA8nxYD5CmpX09-Z7J28Bg1g5n_XM)

### Decision Tree
Decision Trees are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. 

We performed tuning on hyperparameters like 

```shell
max_depth, max_features and min_sample_split
```

![img](https://lh5.googleusercontent.com/Ca7ZXEQ1qw33moSxYSGbCX4gS2Zn1mZuuchlzMG0TsqLpjlsxEb9_BkV6CFcjfXgiwdLMEGlFSHEC6Tru606i2bowMAL0gt9TIfFKwglYhskkre4I8MHD6_SUb4-rAg-ERUtFzMF)

![img](https://lh5.googleusercontent.com/FWKs-_Sk-Q9FP9dUQFUYd5Z1Hzje5utu0eaTaE3L6wqLnXU5m-WV8qWPdRFDUZXA6M2q4MxNia_b0Qo1aggJOGg4Q7kWPt-h_t77HYaDRtPwcwBUHbsDpexe_IjqIU774Z2RE2WG)

### Random Forest

The random forest is a classification algorithm consisting of many decision trees. It tries to create an uncorrelated forest of trees through a bagging process and its overall prediction  is more accurate than that of any individual tree.

We performed tuning on hyperparameters like:

``` max_depth, max_features & min_sample_split ```

![img](https://lh3.googleusercontent.com/PUJ6vbx5cPOIpwncqZnJVMGum1ZdcW_zAn7OAjRx1sePJZZjdztcjUSeSh7pBcXYaia_htsBeAdTrR_fvGwZwVXwQlVh1BbdVFpERaDg3Gf7nmk_zgndMiP_kgGZ8uDluQ_aB94H)

![img](https://lh6.googleusercontent.com/D5jtf3UsAeWzOAE2HL1Gc_o85XQPHm_tLrsv33wywww3jKOwxQmchaa2jub52NDKla_w6ZhT4ekZcWPjzQ97WLwx6iQxzOvl2HXhSr0alb8A9Ob2CodH2RDyUjg_la3dkjrAIxhX)




### Neural Network
Neural networks try to mimic the human brain and solve complex problems by recognizing hidden patterns in data. For our dataset, we have built a 1D Convolutional Neural Network (CNN) with 5 hidden layers having 32, 24, 20, 16, and 4 neurons respectively. We used batch normalization to normalize the hidden layers and used ‘elu’ as our activation function. We also used Average pooling to create downsampled features as input to the next layer.

![img](https://lh5.googleusercontent.com/poOw3gCbXIWVCGFzW8wCGWS190H5PMPJzaOu7I6c2M-OU1z1OzNgLs6qRsJIIxQygROCAiuGolGpyugsOK4p7MjWO79WQIa2vZnt69AKiTCUw-XOP-fs-Ymnc_9fPZ0Adexfnsd3)

![img](https://lh5.googleusercontent.com/Ys5hvls94Ic1wpoV7ItSfCHSZ2TAhzseKvB2zjQO9LlDzoQD5hPmVf7ptOnEUPfhk6K6RJyxGCEL7BfbLxZchq66c2UZzkEsuMv7-LhGISyPZ0ZooLOkXKobOPJgBLF0fGXxCmIe)

## Ensemble Models

### Boosting
#### XGBoost
XGBoost is an optimised gradient boosting library that has been designed for efficiency, flexibility, and portability. It uses the Gradient Boosting framework to create machine learning algorithms. XGBoost is a parallel tree boosting algorithm that solves a variety of data science issues quickly and accurately. To deal with class imbalance in the data, we used.

``` scale pos weight = count of negative classcount of positive class```

```
best params = {‘n_estimators’: 200,
			         ‘max_depth’: 5,
			         ‘learning_rate’: 0.05}
```

**Output:**

![img](https://lh6.googleusercontent.com/FtoIfwQCRNPoRAnqpwXgh751BAGxQaxwi-UEu-o9mRK5cEbCyVFekm48vsZQnC60BCT8erTMUjII4wRUSQdJywVjieZlOMrgQp6iAbnMeVcNiYNGKBjXaKvV-aRTa1QIt5tcXaRN)

![img](https://lh3.googleusercontent.com/R2xvV-wTIS-J3fB6MYs6N5cNOeSN8_wBORmshSBja6ApByzTAU5riCW_PS3dDsZR-w6gNcb8AoBggHIoO2PwCcx8TFhLyA_stpzCvgdCgD_KCTp3S0KMksGbiazPYvunlXL3PTfv)




#### Light GBM
Light GBM is a fast and high-performance gradient boosting algorithm based on decision trees to reduce memory usage and increase efficiency. It splits the tree leaf wise whereas other boosting algorithms split the tree level-wise or depth-wise. 
##### *Best Params:*
```
bagging_fraction=0.331, bagging_freq=5, boost='gbdt',
boost_from_average='false', class_weight='balanced',
feature_fraction=0.0405, learning_rate=0.0083, metric='auc',
min_data_in_leaf=80, min_sum_hessian_in_leaf=10.0, num_leaves=13,
num_threads=8, objective='binary', tree_learner='serial', verbosity=1
```

![img](https://lh3.googleusercontent.com/yd2HQHPMbWYZaWfkN8tiDbz-4Hzvjtf9FGIG_C_zrAHeqc1949Tm0FqZPMktz-I9t5GXtpZIA_-C5affSe773ZfswNDextAMhMT6dbohvhgW_J2u3fV-mw6p3Vv0Jgpq9ipZXCWC)

![img](https://lh6.googleusercontent.com/D7YypGL6zFA-VVpLuFEecqJgGprWU05pMu-meEMMXfOYebRCKm12Okt322xCHfNndA1wtjwd4ngkE-Zo_Tv52FxqTqzuOBF-TSFGWO56mYUTCRFoLpxskHrlF-p55U8Ikl2pTaPO)

![img](https://lh5.googleusercontent.com/VpmmJxcQa9tVvL9tTWlgOikvO9f-D1dXlRyx45PAPyijLw79CIswqcThp3l_E5OUaAR1cid1Z8YWyywwlWgk7-9y2Ld_xZ9b9SflavDGnUFfUbbz3f7YML6CzhlClpnk9dZCbOof)



### Stacking [best model]
The models and their best parameters found in the individual models above were compared and analyzed to select 3 models to build a stacked ensemble model. The following configuration was used:

**Base Estimators** : [LightGBM, Logistic Regression]
**Final Estimator** : [XGBoost Classifier]
#### Estimators
![img](https://lh4.googleusercontent.com/NMsrh8DO7NC6FFNsTLXTWk2nRnTuWPDrehSw7aHqugU8cQbEkk97B4t6z2wfuMLdKBJf8y4zND9_PTaKSiUZJEg8rCq28t7eP6ND8oFAOkqmCwUQmsL7pm8HL4k2BljY-uIzlRHT)

#### Validation Results

![img](https://lh3.googleusercontent.com/QymSFPurT7ucyD8JcAg3STDnMrrJAIx4B2tXoMLAUak4kksCFwkEvVuUbb-TUKOhPNxbaEQRaisb5j3C8QlqpZYYLRyURUdMJ8mDuQ1Hul2-MJyHUEGQShvkvDfkN3TbPDHea8kS)

![img](https://lh3.googleusercontent.com/E0ggwsYIcdgnNfnsRFvxmdYMONGEV35jDXF8tV2-XMIf7jlhI7NyOJnwJkAEhJY2kInVFup6cYx7H3-becpg4ksakOAhrR9C2F0fMUUY6lCeZIMnnd2SENEx6bcFO1WqVM44gmbl)

#### Test Result on Kaggle competition
![img](https://lh4.googleusercontent.com/p_c2VKIfmMc0mhIAav7WnTLcfvUBeVljcGFY2A78HTQBU6CiHScFBLT152cDuRj0-F1O-3IhWm0INClQ-cPiBfP97gI1CRjxxPma_chJHT3y3Z2OWx4wOBrCofSguvOFJMr9Obf6)
# Analysis
The models are compared based on a base performance metric - Area under the Curve.

## Methodology - AuC
The AuC-RoC curve is a great performance measure for classification problems.

When we need to check or visualize the performance of the multi-class classification problem, we use the AUC (Area Under The Curve) ROC (Receiver Operating Characteristics) curve. It is also written as AUROC (Area Under the Receiver Operating Characteristics).

The ROC curve is plotted with the True Positive Rate (TPR) against the False Positive Rate (FPR) where TPR is on the y-axis and FPR is on the x-axis.

``` TPR = TP/TP + FN ```                       
 ``` FPR =FP / FP + TN ```


![img](https://lh5.googleusercontent.com/FH1uVtigGOo1CluNczZgk91elhT9SJ_9axNI6eGcfLG0UXNg2t1p7FekPRoan5oe-s2AFj0GMxWpNU0E7xFeleyU64wgDnavqrB4zaEMBTAa4dYVcx2-tLChPTsZtoOOSNLLyDhZ)
## Model Comparison
The AuC scores of all models were collated and compared to see which model is giving us the best performance. In our scenario, we find out that Logistic Regression, XGBoost Classifier and LightGBM are our top 3 performing models.

We proceeded to choose these three models to build our stacked ensemble model. The ensemble model built resulted in a slightly better performance than the individual models. Overall, stacking the models increased our AuC by ~1%.

![img](https://lh5.googleusercontent.com/HJsgvKLt_6opkBhquaxVHIkVklwAivk7OT1agvihvs55hBp1NouTmJBamzFShsl9XwbsAEpsIv_dxch9-n0WNfpxdlSJYds-l0zjbi5M66Dazai_Q8KKwx80cc5X4fOKgSy6b1UY)

![Chart](https://lh6.googleusercontent.com/XOU3dFblQCJMDIgbnoIz-o_qOr0Wz1DC_E_-Xj2Gvsziw5jLXTc6g6TC1sMuwmeJPbBV-ZvjNmvgwaU1u8rvjeUUZlEb4VV_cs6_7IrTUKBpmHRKPnItZy7D5i2ZxP6OyQ)


# Summary
**EDA:** Data was explored and the presence of noise and synthetic data was noticed. Added 600 new variables on top of the existing 200 to address the synthetic data issue.

**Model Comparison:** All individual models were compared based on the AuC under RoC. The 3 best performing models on validation data were used to create a stacking ensemble model.

**Final model used for classification:** Stacking ensemble models using LightGBM and logistic regression models as base estimators, and XGBoost model as the final estimator

**AUC obtained using the final model:** 0.89 
## Cost Analysis
*Source of reference for cost benchmarks: [DataSource*](https://archive.ics.uci.edu/ml/datasets/Statlog+\(German+Credit+Data\))

Taking the benchmarks from a German-based banking data and using those benchmarks in our model, we assume that the cost of misclassifying a customer who would have actually made a transaction is 5 times the cost of misclassifying a customer who would not have made the transaction.

Based on the confusion matrix we obtained above, our misclassification cost is:

``` 758\*5 + 7,406\*1 = $11,196 ```

## References
- <https://www.kaggle.com/code/nawidsayed/lightgbm-and-cnn-3rd-place-solution/notebook>
- <https://www.kaggle.com/code/mks2192/list-of-fake-samples-and-public-private-lb-split/notebook>
- <https://github.com/tianqwang/Santander-Customer-Transaction-Prediction>

