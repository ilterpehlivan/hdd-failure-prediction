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

This data set represents snapshot of each operational hard drive in the Backblaze data center.It includes basic drive information along with the [S.M.A.R.T.](https://en.wikipedia.org/wiki/S.M.A.R.T.) statistics reported by that drive. The daily snapshot of one drive is one record or row of data. All of the drive snapshots for a given day are collected into a file consisting of a row for each active hard drive. The format of this file is a "csv" (Comma Separated Values) file. Each day this file is named in the format YYYY-MM-DD.csv, for example, 2013-04-10.csv.

The first row of the each file contains the column names, the remaining rows are the actual data. 

This dataset contains data from the first two quarters in 2016.

The columns are as follows:

**Date** – The date of the file in yyyy-mm-dd format

**Serial Number** – The manufacturer-assigned serial number of the drive

**Model** – The manufacturer-assigned model number of the drive

**Capacity** – The drive capacity in bytes.

**Failure** – Contains a “0” if the drive is OK. Contains a “1” if this is the last day the drive was operational before failing.

**90 variables that begin with 'smart'** - Raw and Normalized values for 45 different SMART stats as reported by the given drive


## pre-analysis results
* This is supervised ML problem with classification (Success/Failure)
* There are more success cases than failure, that makes dataset unbalanced, this is based on the following mean-failure/model diagram; <br/>

![alt text](https://github.com/ilterpehlivan/hdd-failure-prediction/blob/master/Figure_1.png?raw=true "Medium Failure Graph")

* Dataset is too big 2.5 GB, that makes it super hard to process
* 90 variables takes quite long time to train and potential overfitting

## Feature Selection

Due to limited RAM on our personal computers, we limited our feature size to at most 10 SMART attributes <br/>
We chose to keep BackBlaze’s original five features SMART 5, 187, 188, 197, and 198 because they were selected by BackBlaze for their high correlation to HDD failure<br/>
Additionally we put the SMART 9 which is the lifetime of the HDD as we believed it should be correlated to failures

## Preprocessing

* Choose raw data over normalized
>Selecting both normalized and raw is redundant as they both represent the same data points.How the normalization applied is not clear that is why those columns filtered

* Balance the dataset
>The success rates (0s) are more than failure rate (1s).That makes the dataset an unbalanced dataset. It is illustrated in this figure 

>To balance the data firstly oversampling is applied to the lower sampled (1) classes then undersampling is applied for the high rate of classification(0s) and finally Kfolding is used to select the random sub-groups for train/test set.

## Algorithms

### Logical Regression

TODO: < Khaled >

### Naive Bayes
TODO: #Phase2

### Random Forest
TODO: #Phase2

## Improvements
* Focus to the certain model which has more failure rates
* Download more data from the site and select only focused model data
* Try to find other features which might be correlated
* Try different algorithms
