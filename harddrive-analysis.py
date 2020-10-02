
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from numpy import mean
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_score, recall_score, accuracy_score, f1_score, auc, confusion_matrix, \
    zero_one_loss, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

DATA_SET_PATH = "./data/harddrive.csv"

# NOT PART OF THE RUNNING CODE JUST FOR STATISTICS
# First we first check some statistics
def get_hdd_statistics(df):
    print(df.head(5))
    # number of hdd
    print("number of hdd:", df['serial_number'].value_counts().shape)

    # number of different types of harddrives
    print("number of different harddrives", df['model'].value_counts().shape)

    group_by_model = df.groupby("model")["failure"]
    mean_failure = group_by_model.mean()
    print(mean_failure.describe)
    group_by_model_agg = group_by_model.agg(
        ['count', 'sum', 'mean']
        # sum=pd.NamedAgg(column="failure", aggfunc="sum"),
        # count=pd.NamedAgg(column="failure", aggfunc="count")
    )
    group_by_model_agg["fail_rate"] = group_by_model_agg["mean"]*(10**5)
    group_by_model_agg.to_csv("agg_results.txt")
    # print("highest rate failed model:",df.loc[df.model == "WDC WD3200BEKT"][["model","serial_number","failure"]])
    # ax = mean_failure.plot(kind='bar', figsize=(10, 6), color="indigo", fontsize=13);
    ax = mean_failure.plot.bar()
    ax.set_title("Distribution Diagram", fontsize=22)
    ax.set_ylabel("Failure mean percentage", fontsize=15)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Model")
    plt.show()

# NOT PART OF THE RUNNING CODE JUST FOR STATISTICS
def eval_basic_logical_regression_kfold(features, df):
        """
        Evaluating Logical Regression
        :param features: selected features
        :param df: dataframe
        :return: instance of EvalIndex,it includes four attribute:precission,recall,accuracy,f1_score
        """
        roc_auc, precision, recall, acc, f1, auc_score = [[] for _ in range(6)]

        X, y = init_model(df, features)
        # trying to fix scewness
        X = np.log1p(X)

        # print("X->", X)
        # print("y->", y)

        # split into train/test sets %70 train and %30 test
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        # fit a model
        model = LogisticRegression()
        # apply cross validation i.e K-Fold
        k = 5
        kf = KFold(n_splits=k, shuffle=True, random_state=1)
        for train_indecies, test_indecies in kf.split(X):
            X_train = X.iloc[train_indecies]
            X_test = X.iloc[test_indecies]
            y_train = y[train_indecies]
            y_test = y[test_indecies]
            # print the split rates
            print_split_rates(y_train, y_test)

            print("\nThe model fits the TRAIN SET for train and test")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # cumulative Scores
            precision += [precision_score(y_test, y_pred, average='binary')]
            recall += [recall_score(y_test, y_pred, average='binary')]
            acc += [accuracy_score(y_test, y_pred)]
            f1 += [f1_score(y_test, y_pred, average='binary')]
            # auc_score += [roc_auc_score(y_test,y_pred)]
            # cross_val_score()

            # print ROC curve
            fpr, tpr, thresholds = roc_curve(y_test, y_pred)
            roc_auc += [auc(fpr, tpr)]

            # print confusion matrix
            modelResults = confusion_matrix(y_test, y_pred)
            error = zero_one_loss(y_test, y_pred)
            print("Results for The model:\n")
            print(modelResults)
            print("Accuracy:", 1 - error)
            # print(1 - error, '\n')
            print(classification_report(y_test, y_pred))

        roc_auc = sum(roc_auc) / k
        print("\nprecision:{0}\nrecall:{1}\naccuracy:{2}\nf1_score:{3}".format(sum(precision) / k, sum(recall) / k,
                                                                               sum(acc) / k, sum(f1) / k))


        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        # plt.savefig("ROC.png", dpi=300)

# ACTUAL MODEL TRAINING START HERE
def clean_not_failing_models(df):
    #First drop constant columns
    print(df.shape)
    df = df.loc[:, ~df.isnull().all()]
    print(df.shape)

    # number of different types of harddrives
    print("before cleanup->number of different harddrives", df['model'].value_counts().shape)
    group_by_model = df.groupby("model")["failure"]
    # mean_failure = group_by_model.mean()
    # print(mean_failure.describe)
    group_by_model_agg = group_by_model.agg(
        ['count', 'sum', 'mean']
    ).reset_index()
    group_by_model_agg["fail_rate"] = group_by_model_agg["mean"] * (10 ** 5)

    zero_failed_models = group_by_model_agg.loc[group_by_model_agg["fail_rate"] == 0.0]
    print("**Print the HDD models which are not failing at all**")
    print(zero_failed_models)
    # print(type(zero_failed_models))
    #print(list(zero_failed_models.columns.values))
    # print(zero_failed_models.iloc[:,0])

    # lets drop the zero failed models from df
    df_cleaned = df.loc[~df['model'].isin(zero_failed_models["model"])]

    print("after cleanup->number of different harddrives", df_cleaned['model'].value_counts().shape)
    return df_cleaned

def print_split_rates(y_train, y_test):
    # summarize train and test composition
    train_0, train_1 = len(y_train[y_train == 0]), len(y_train[y_train == 1])
    test_0, test_1 = len(y_test[y_test == 0]), len(y_test[y_test == 1])
    print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))


def eval_with_sampling_and_kfold_logical_regression(features, df,k=5):
    """
    evaluating logical regression with over+under sampling to find the balanced weights
    and applying Kfold CV over it
    :param features: selected features
    :param df: dataframe from the dataset
    :param k: kfold value for Kfold CV
    :return: precission,recall,accuracy,f1_score
    """
    roc_auc, precision, recall, acc, f1, auc_score = [[] for _ in range(6)]

    X, y = init_model(df, features)
    # trying to fix scewness
    X = np.log1p(X)

    print("**Before sampling**")
    # print("X->", X.describe())
    print("y-Mean", mean(y))

    # split into train/test sets %70 train and %30 test
    X, X_test_origin, y, y_test_origin = train_test_split(X, y, test_size=0.3)
    # fit a model
    model = LogisticRegression()
    over = SMOTE(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('over', over), ('under', under)]
    pipeline = Pipeline(steps=steps)
    # apply cross validation i.e K-Fold
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
    X,y = pipeline.fit_resample(X,y)

    print("**After sampling**")
    # print("X->", X.describe())
    print("y-Mean", mean(y))

    # enumerate the splits and summarize the distributions
    for train_ix, test_ix in kfold.split(X, y):
        X_train = X.iloc[train_ix]
        X_test = X.iloc[test_ix]
        y_train = y[train_ix]
        y_test = y[test_ix]
        # print the split rates
        print_split_rates(y_train, y_test)

        print("running the model to fit..")
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        # Scores
        precision += [precision_score(y_test, y_pred, average='binary')]
        recall += [recall_score(y_test, y_pred, average='binary')]
        acc += [accuracy_score(y_test, y_pred)]
        f1 += [f1_score(y_test, y_pred, average='binary')]
        # auc_score += [roc_auc_score(y_test,y_pred)]
        # cross_val_score()

        # collect for ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc += [auc(fpr, tpr)]

        #print confusion matrix
        modelResults = confusion_matrix(y_test, y_pred)
        error = zero_one_loss(y_test, y_pred)
        print("Results for The model:\n")
        print(modelResults)
        print("Accuracy:",1 - error)
        # print(1 - error, '\n')
        print(classification_report(y_test, y_pred))

    roc_auc = sum(roc_auc) / k
    print("*****************\nSummary results of the sampled train/test:\nprecision:{0}\nrecall:{1}\naccuracy:{2}\nf1_score:{3}".format(sum(precision) / k, sum(recall) / k,
                                                                           sum(acc) / k, sum(f1) / k))
    plt.title('Sampled - Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC-Sampled = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # plt.show()
    # plt.savefig("Sampled-LogR-ROC.png", dpi=300)

    # Results based on original test data (not sampled)
    y_pred_final = model.predict(X_test_origin)
    # Scores
    modelResults_f = confusion_matrix(y_test_origin, y_pred_final)
    error_f = zero_one_loss(y_test_origin, y_pred_final)

    print("*****************\nFinal Results for The model:")
    print(modelResults_f)
    print("Final Accuracy:",1 - error_f)
    # print(1 - error_f, '\n')
    print(classification_report(y_test_origin, y_pred_final))
    # print("\nprecision:{0}\nrecall:{1}\naccuracy:{2}\nf1_score:{3}".format(modelResults_f, sum(recall) / k,
    #                                                                        sum(acc) / k, sum(f1) / k))
    # Print the final auc
    fpr_final, tpr_final, thresholds_final = roc_curve(y_test_origin, y_pred_final)
    roc_auc_final = auc(fpr_final, tpr_final)
    print("**Printing the final auc ***")
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr_final, tpr_final, 'b', label='AUC-Origin = %0.3f' % roc_auc_final)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("Origin-LogR-ROC.png", dpi=300)






# This function common for other algorithms
def init_model(df, features):
    features_specified = []
    for feature in features:
        features_specified += ["smart_{0}_raw".format(feature)]
    df = df.fillna(0)
    X = df[features_specified]
    y = df['failure'].values
    return X,y


def print_some_info_about_dataset(df):
    print(df.head(10))
    print(df.info())
    # print(df.columns.values)
    # print(len(df.columns.values))

def main():
    # plot_fail_per_model()
    df = pd.read_csv(DATA_SET_PATH)
    print_some_info_about_dataset(df)
    cleaned_df = clean_not_failing_models(df)
    # get_hdd_statistics(df)
    #This is an unbalanced data
    # features = [5, 9, 187, 188, 193, 194, 197, 198, 241, 242]
    features = [5, 9, 187, 188, 197, 198]
    # eval_basic_logical_regression_kfold(features, cleaned_df)
    eval_with_sampling_and_kfold_logical_regression(features,cleaned_df)

if __name__ == '__main__':
    main()
