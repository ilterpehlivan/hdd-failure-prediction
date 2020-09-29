# hdd-failure-prediction
## Introduction
In data center environments, hard disk drive (HDD) failures are a rare but costly occurrence.
Currently, HDD manufacturers use Self-Monitoring and Reporting Technology (SMART) attributes collected during normal
operations to predict failures. SMART attributes represent HDD health statistics such as the number of scan errors,
reallocation counts and probational counts of a HDD. If a certain attribute considered critical to HDD health goes
above its threshold value, the HDD is marked as likely to fail

This project aims to find the best model to predict the failures by using different ML algorithms

For this case study following dataset is used
https://www.kaggle.com/backblaze/hard-drive-test-data

## Dataset

TODO: < Deval >

## Feature Selection

TODO: < Ilter >

## Preprocessing

* Choose raw data over normalized
>Selecting both normalized and raw is redundant as they both represent the same data points.How the normalization applied is not clear that is why those columns filtered

* Balance the dataset
>The success rates (0s) are more than failure rate (1s).That makes the dataset an unbalanced dataset. It is illustrated in this figure 

![Model Distribution](Figure_1.jpg?raw=true "Model failure rates")

>To balance the data firstly oversampling is applied to the lower sampled (1) classes then undersampling is applied for the high rate of classification(0s) and finally Kfolding is used to select the random sub-groups for train/test set.

## Algorithms

### Logical Regression

TODO: < Khaled >

### Naive Bayes
TODO: #Phase2

### Random Forest
TODO: #Phase2

## Improvements
TODO: < ILTER > explain how to improve the results
