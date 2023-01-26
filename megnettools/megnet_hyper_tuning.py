"""
Module for tuning the hyperparameters of a MEGNet model.
The module includes a single function `HyperTuningMEGNetModel` which takes in featurized
data and structure data, as well as hyperparameter lists and train-test split ratio.
The function returns a dataframe with the hyperparameters of the model and MAE values."""
from .megnet_setup_evaluate import megnet_regression_model, generate_model_scaler
import pandas as pd
import os
from pickle import dump, load
from sklearn.model_selection import train_test_split
from typing import List

def megnet_train_val_scores(model,scaler,X_train,y_train,X_test,y_test,savedir='./'):
    # now get MAEs
    scores_dict = megnet_regression_model(
        model, scaler, X_test, y_test,  id=idf+'_val', savedir=savedir)
    # getting MAE train
    train_scores_dict = megnet_regression_model(
        model,  scaler, X_train, y_train,id=idf+'_train', savedir=savedir)    
    train_scores_dict = {f"{k}_train": v for k, v in train_scores_dict.items()}
    scores_dict.update(train_scores_dict)
    return scores_dict
 

def hyper_tuning_megnet_model(featurized_data: pd.DataFrame, structure: pd.DataFrame,
                              batch_sizes: List[int] = [16, 32, 64],
                              epochs: List[int] = [50, 100, 200],
                              neuron_layers: List[int] = [(64,32,16)],
                              train_test_split_ratio: float = 0.1,
                              random_state_split: int = 1,
                              prefix_name: str = 'hypertuning_megnet_model',
                              **kwargs) -> pd.DataFrame:
    """
    This function is used to tune the hyperparameters of a MEGNet model. The function starts by
    splitting the data into train and test sets, then it trains the model on different combinations
    of hyperparameters (batch size, epochs, n1, n2, n3) and saves the model along with the scaler
    and calculates the MAE for both train and test sets.
    The function returns a dataframe with hyperparameters of the model and MAE values.
    """

    # feat_data and structure will be the train split from now on.
    savedir = kwargs.get("modeldir", prefix_name+"_models/")
    try:
        os.makedirs(savedir)
    except FileExistsError:
        pass
    # Initialize an empty dataframe
    results_df = pd.DataFrame()

    structures, structures_test, train_data, test_data = train_test_split(
        structure, featurized_data, test_size=train_test_split_ratio,
        random_state=random_state_split)
    # define loops
    for batch_size in batch_sizes:
        for epoch in epochs:
            for (n1,n2,n3) in neuron_layers:
                model, scaler = generate_model_scaler(
                    train_data,
                    structures,
                    epochs=epoch,
                    batch_size=batch_size,
                    ntarget=len(train_data.columns),
                    n1=n1,
                    n2=n2,
                    n3=n3,
                    save_model=False
                )
                idf = prefix_name + \
                    f"bs{batch_size}_e{epoch}_n1n2n3_{n1}x{n2}x{n3}"
                model.save_weights(savedir+idf+"_weights.h5")
                dump(scaler, open(savedir+idf+'_scaler.pkl', 'wb'))
                ## get score dict
                settings_dict={'batch_size': batch_size,
                                'epoch': epoch,
                                'n1': n1,
                                'n2': n2,
                                'n3': n3,
                                }
                score_dict=megnet_train_val_scores(model,scaler,structures,train_data,
                                        structures_test,test_data,savedir=savedir)
                # Append the current run's results to the dataframe
                results_dict=settings_dict.update(score_dict)                
                results_df = results_df.append(results_dict, ignore_index=True)
                results_df.to_csv("results_tmp.csv", sep='\t', index=False)
    results_df.to_csv("results.csv", sep='\t', index=False)


__all__ = ['HyperTuningMEGNetModel']
