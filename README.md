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

This data set represents snapshot of each operational hard drive in the Backblaze data center.It includes basic drive information along with the S.M.A.R.T. statistics reported by that drive. The daily snapshot of one drive is one record or row of data. All of the drive snapshots for a given day are collected into a file consisting of a row for each active hard drive. The format of this file is a "csv" (Comma Separated Values) file. Each day this file is named in the format YYYY-MM-DD.csv, for example, 2013-04-10.csv.

The first row of the each file contains the column names, the remaining rows are the actual data. The columns are as follows:

Date – The date of the file in yyyy-mm-dd format.
Serial Number – The manufacturer-assigned serial number of the drive.
Model – The manufacturer-assigned model number of the drive.
Capacity – The drive capacity in bytes.
Failure – Contains a “0” if the drive is OK. Contains a “1” if this is the last day the drive was operational before failing.
2013-2014 SMART Stats – 80 columns of data, that are the Raw and Normalized values for 40 different SMART stats as reported by the given drive. Each value is the number reported by the drive.
2015-2017 SMART Stats – 90 columns of data, that are the Raw and Normalized values for 45 different SMART stats as reported by the given drive. Each value is the number reported by the drive.
2018 (Q1) SMART Stats – 100 columns of data, that are the Raw and Normalized values for 50 different SMART stats as reported by the given drive. Each value is the number reported by the drive.
2018 (Q2) SMART Stats – 104 columns of data, that are the Raw and Normalized values for 52 different SMART stats as reported by the given drive. Each value is the number reported by the drive.
2018 (Q4) SMART Stats – 124 columns of data, that are the Raw and Normalized values for 62 different SMART stats as reported by the given drive. Each value is the number reported by the drive.


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
