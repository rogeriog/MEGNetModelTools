from MEGNetSetupEvaluate import setup_threading, model_setup, regression_model, load_CustomModel, \
                                   get_standard_scaler, generate_model_scaler, load_model_scaler
import pandas as pd
import numpy as np
import os 
from pickle import dump, load
import time, shutil
from sklearn.model_selection import train_test_split
from typing import List

def HyperTuningMEGNetModel(featurized_data: pd.DataFrame, structure: pd.DataFrame,
    batch_sizes: List[int] = [16,32,64],
    epochs: List[int] = [50,100,200],
    n1s: List[int] = [64,128],
    n2s: List[int] = [32,128,256],
    n3s: List[int] = [16,64,128],
    train_test_split_ratio: float = 0.1,
    random_state_split : int = 1,
    prefix_name: str = 'HypertuningMEGNetModel',
    **kwargs) -> pd.DataFrame:
    """
    This function is used to tune the hyperparameters of a MEGNet model. The function starts by splitting the data into train and test sets, then it trains the model on different combinations of hyperparameters (batch size, epochs, n1, n2, n3) and saves the model along with the scaler and calculate the MAE for both train and test sets.
    The function returns a dataframe with hyperparameters of the model and MAE values.
    """

    ### feat_data and structure will be the train split from now on.
    savedir=kwargs.get("modeldir",prefix_name+"_models/")
    try:
        os.makedirs(savedir)
    except FileExistsError:
        pass
    # Initialize an empty dataframe
    results_df = pd.DataFrame(columns=['batch_size', 'epoch', 'n1', 'n2', 'n3', 'MAE_test', 'MAE_train'])

    structures, structures_test, train_data, test_data = train_test_split(
                        structure, featurized_data, test_size=train_test_split_ratio, 
                        random_state=random_state_split)
    #define loops
    for batch_size in batch_sizes:
        for epoch in epochs:
            for n1 in n1s:
                for n2 in n2s:
                    for n3 in n3s:
                        model,scaler = generate_model_scaler(
                                                    train_data,
                                                    structures,
                                                    epochs=epoch,
                                                    batch_size=batch_size,
                                                    ntarget=len(train_data.columns), 
                                                    n1 = n1,
                                                    n2 = n2,
                                                    n3 = n3,
                                                    save_model=False
                                                    )
                        idf=prefix_name+f"bs{batch_size}_e{epoch}_n1n2n3_{n1}x{n2}x{n3}"
                        model.save_weights(savedir+idf+"_weights.h5")
                        dump(scaler, open(savedir+idf+'_scaler.pkl', 'wb'))
                        ### now get MAEs
                        X=structure_test
                        y=test_data
                        MAE_test = regression_model(model,X,y,scaler,id=idf+'_val',savedir=savedir)
                        ### getting MAE train
                        X=structure
                        y=train_data
                        MAE_train = regression_model(model,X,y,scaler,id=idf+'_train',savedir=savedir)
                        # Append the current run's results to the dataframe
                        results_df = results_df.append({'batch_size': batch_size,
                                                       'epoch': epoch,
                                                       'n1': n1,
                                                       'n2': n2,
                                                       'n3': n3,
                                                       'MAE_test': MAE_test,
                                                       'MAE_train': MAE_train}, ignore_index=True)
    results_df.to_csv("results.csv",sep='\t', index=False)

