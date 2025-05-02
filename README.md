# Model Evaluation Results

This document provides a detailed analysis of the results from evaluating various machine learning models on different balanced datasets. The primary objective of this evaluation was to assess the performance of each model in predicting customer status, with a particular focus on addressing the challenges posed by imbalanced data.

## Key Observations

* **Top-Performing Models:** XGBoost and CatBoost consistently demonstrated superior performance across most resampling techniques. These models achieved high accuracy (approximately 84-85%) and exhibited a strong balance between precision, recall, and F1-score for both classes (Stayed and Left). Their robustness across different resampling methods suggests that they are well-suited to handle the complexities of this dataset.
* **Impact of Resampling:** The choice of resampling technique significantly influenced model performance. Techniques such as SMOTE and ADASYN generally improved the performance of models like Logistic Regression by mitigating the class imbalance. However, the effectiveness of resampling varied across different models, highlighting the importance of selecting an appropriate resampling strategy for each algorithm.
* **AdaBoost Limitations:** AdaBoost exhibited poor performance, particularly in conjunction with the ADASYN resampling technique, achieving an accuracy of only 26%. This suggests that AdaBoost may not be suitable for this specific problem or requires substantial hyperparameter tuning to achieve acceptable results. Its sensitivity to certain resampling methods warrants careful consideration in future applications.
* **Challenges of Class Imbalance:** Even with the application of resampling techniques, accurately predicting the minority class ('Stayed' - class 0) remained challenging for several models. This is a common issue in imbalanced datasets, where models tend to be biased towards the majority class. Achieving a high precision and recall for the minority class often requires a trade-off and careful selection of evaluation metrics.
* **Logistic Regression Variability:** Logistic Regression's performance was highly dependent on the resampling technique employed. While it generally exhibited higher recall for the minority class, this often came at the cost of lower precision, indicating a tendency to classify more instances as 'Stayed' but with a higher rate of false positives.
* **KNN Underperformance:** The K-Nearest Neighbors (KNN) algorithm generally performed the weakest among the evaluated models. Its performance was limited across all resampling techniques, suggesting that it may not be effective in capturing the underlying patterns in this dataset.

## Models Evaluated

The following models were included in the evaluation:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* AdaBoost
* K-Nearest Neighbors (KNN)
* Naive Bayes
* XGBoost
* LightGBM
* CatBoost

## Resampling Techniques

The following resampling techniques were used to address the class imbalance in the dataset:

* SMOTE
* ADASYN
* Random Oversampler
* BorderlineSMOTE
* Random Undersampler
* Tomek Links

## Detailed Results and Analysis

To provide a clearer picture of model performance, the following sections present a more detailed analysis, focusing on the key models and their performance across different resampling techniques.

### 1.  XGBoost

XGBoost consistently delivered strong results, demonstrating its ability to handle imbalanced data effectively.  Here's a summary of its performance:

| Resampling Technique   | Accuracy (%) | Precision (Class 0) | Recall (Class 0) | F1-Score (Class 0) | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| :----------------------- | :-----------: | :------------------: | :---------------: | :------------------: | :------------------: | :---------------: | :------------------: |
| SMOTE                  |      84.7     |        85.7          |       97.2        |        91.1          |        78.6          |       45.1        |        57.3          |
| ADASYN                 |      84.3     |        85.5          |       97.1        |        90.9          |        77.6          |       43.7        |        55.8          |
| Random Oversampler       |      82.1     |        83.7          |       96.4        |        89.6          |        70.3          |       36.8        |        48.3          |
| BorderlineSMOTE          |      83.9     |        85.1          |       97.0        |        90.6          |        76.5          |       42.3        |        54.4          |
| Random Undersampler      |      77.1     |        79.9          |       91.8        |        85.4          |        56.7          |       28.5        |        38.0          |
| Tomek Links            |      83.0     |        84.3          |       96.6        |        90.0          |        72.7          |       39.2        |        50.8          |

XGBoost maintains high accuracy and a good balance of precision and recall, even with different resampling methods.  It is particularly strong in correctly identifying the majority class (0).

### 2.  CatBoost

CatBoost also shows robust performance across various resampling techniques, very similar to XGBoost:

| Resampling Technique   | Accuracy (%) | Precision (Class 0) | Recall (Class 0) | F1-Score (Class 0) | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| :----------------------- | :-----------: | :------------------: | :---------------: | :------------------: | :------------------: | :---------------: | :------------------: |
| SMOTE                  |     84.9      |       86.0           |       97.1        |       91.2           |       79.2           |       46.0        |       58.3           |
| ADASYN                 |     84.5      |       85.7           |       97.1        |       91.0           |       78.1           |       44.5        |       56.5           |
| Random Oversampler       |     82.3      |       83.9           |       96.4        |       89.7           |       70.9           |       37.4        |        49           |
| BorderlineSMOTE          |     84.1      |       85.3           |       97.0        |       90.7           |       77.1           |       43.0        |       55.1           |
| Random Undersampler      |     77.5      |       80.2           |       92.0        |       85.7           |       57.8           |       29.3        |       39.0           |
| Tomek Links            |     83.2      |       84.5           |       96.7        |       90.2           |       73.3           |       39.8        |       51.5           |

CatBoost's performance is very similar to XGBoost, showing its effectiveness in handling the imbalanced classification problem.

### 3.  Logistic Regression

Logistic Regression's performance varies significantly depending on the resampling method:

| Resampling Technique   | Accuracy (%) | Precision (Class 0) | Recall (Class 0) | F1-Score (Class 0) | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| :----------------------- | :-----------: | :------------------: | :---------------: | :------------------: | :------------------: | :---------------: | :------------------: |
| SMOTE                  |     78.1      |        82.7          |       90.8        |        86.5          |        60.5          |       38.3        |        46.8          |
| ADASYN                 |     78.0      |        82.6          |       90.7        |        86.4          |        60.3          |       38.1        |        46.6          |
| Random Oversampler       |     76.4      |        81.4          |       89.9        |        85.4          |        56.7          |       34.4        |        42.8          |
| BorderlineSMOTE          |     77.6      |        82.2          |       90.5        |        86.1          |        59.2          |       37.0        |        45.5          |
| Random Undersampler      |     74.9      |        78.4          |       93.3        |        85.2          |        52.1          |       24.4        |        33.3          |
| Tomek Links            |     77.2      |        81.9          |       90.2        |        85.8          |        58.1          |       35.7        |       44.4           |

Logistic Regression's performance is highly dependent on the resampling technique.  Oversampling methods (SMOTE, ADASYN) generally perform better than undersampling.  It shows a trend of higher recall for the minority class (1) but lower precision.

### 4. AdaBoost

AdaBoost performed poorly in this evaluation:

| Resampling Technique   | Accuracy (%) | Precision (Class 0) | Recall (Class 0) | F1-Score (Class 0) | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| :----------------------- | :-----------: | :------------------: | :---------------: | :------------------: | :------------------: | :---------------: | :------------------: |
| SMOTE                  |     79.6      |        81.7          |       96.8        |       88.6           |       65.2           |       23.5        |       34.6           |
| ADASYN                 |      26.0      |        56.7          |       43.7        |       49.4           |       43.2           |       56.3        |       49.0           |
| Random Oversampler       |     77.0      |        81.7          |       92.3        |       86.6           |       55.6           |       21.2        |       30.6           |
| BorderlineSMOTE          |     78.7      |        81.2          |       96.3        |       88.1           |       61.5           |       22.1        |       32.6           |
| Random Undersampler      |     74.4      |        77.9          |       93.1        |       84.8           |       51.0           |       23.3        |       32.1           |
| Tomek Links            |      77.7      |        81.3          |       92.7        |       86.6           |       56.8           |       22.0        |       31.6           |

AdaBoost's performance is poor across the board.  Its performance with ADASYN is particularly bad.

## Conclusion

XGBoost and CatBoost are the top-performing models.  The choice of resampling technique is important, and some techniques are better suited to particular models.  AdaBoost performed poorly.  Even with resampling, accurately predicting the minority class remains a challenge.

## Further Work

Further analysis could include:

* Hyperparameter tuning for XGBoost and CatBoost.
* More detailed analysis of the impact of different resampling methods on each model.
* Evaluation of other metrics, such as AUC-ROC and AUC-PR.
* Investigation of alternative models or ensemble methods.