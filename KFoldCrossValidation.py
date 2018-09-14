from random import sample
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Shuffle dataframe rows
def shuffle_rows(df):    
    new_df = df.iloc[sample(range(df.shape[0]),df.shape[0]),:]
    return new_df

# Break dataframe into k subsets and shuffle it
def K_subsets(df, k):
    
    k_datasets = list(range(k))
    window_start = 0
    window_end = round(df.shape[0]/k)
    
    shuffle_data = shuffle_rows(df)
    
    for i in range(k):
        if i == (k-1):
            k_datasets[i] = shuffle_data.iloc[window_start:, :]
        else:
            k_datasets[i] = shuffle_data.iloc[window_start:window_end, :]
            window_start = window_end
            window_end += round(df.shape[0]/k)
          
    return k_datasets

# Take in the a list of dataframes and peform K fold cross validation
# Compute MSE for each iteration of K-CV
# Function assumes one column in the data is named 'y'

def K_regs(K_data):
    k = len(K_data)
    k_MSE = []
    samp_size = []
    
    for i in range(k):
        #remove one of the k data sets
        k_data_temp = [K_data[j] for j in range(k) if j != i]
        df_train_data = pd.concat(k_data_temp)
        
        #create all the training and test sets
        X_train = pd.DataFrame(df_train_data.drop('y', axis = 1))
        Y_train = pd.DataFrame(df_train_data.loc[:,'y'])
        X_test = pd.DataFrame(K_data[i].drop('y', axis = 1))
        Y_test = pd.DataFrame(K_data[i].loc[:,'y'])
        
        #run the regression and get the predicted values
        lr = LinearRegression()
        lr.fit(X_train, Y_train)
        y_pred = lr.predict(X_test)
        
        #calculate MSE and store it in K_preds
        difference_y = np.array(y_pred) - np.array(Y_test)
        difference_squared = difference_y ** 2
        MSE = np.divide(np.sum(difference_squared), X_test.shape[0])
        samp_size.append(X_test.shape[0])
        k_MSE.append(MSE)
        
    return k_MSE, samp_size 

# Calculate overall MSE for the K-folds
def combine_MSE(MSE, Sample_Sizes):
    squared_error = np.array(MSE) * np.array(Sample_Sizes)
    summed_sqerror = np.sum(squared_error)
    overall_MSE = np.divide(summed_sqerror, sum(Sample_Sizes))
    
    return overall_MSE

# Final Function for K-fold cross validation
def K_cross_val(df, k):
    Kfolds_data = K_subsets(df, k)
    results_mse, results_samp = K_regs(Kfolds_data)
    mse_final = combine_MSE(results_mse, results_samp)
    
    return mse_final
    