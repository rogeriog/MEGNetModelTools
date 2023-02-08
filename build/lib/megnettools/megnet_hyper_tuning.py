"""
Module for tuning the hyperparameters of a MEGNet model.
The module includes a single function `HyperTuningMEGNetModel` which takes in featurized
data and structure data, as well as hyperparameter lists and train-test split ratio.
The function returns a dataframe with the hyperparameters of the model and MAE values."""
from .megnet_setup_evaluate import megnet_train_val_scores, generate_model_scaler
import pandas as pd
import os
from pickle import dump, load
from sklearn.model_selection import train_test_split
from typing import List

def hyper_tuning_megnet_model(featurized_data: pd.DataFrame, structure: pd.DataFrame,
                              batch_sizes: List[int] = [16, 32, 64],
                              epochs: List[int] = [50, 100, 200],
                              learning_rates: List[float] = [0.001],
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
            for learning_rate in learning_rates:
                for (n1,n2,n3) in neuron_layers:
                    model, scaler = generate_model_scaler(
                        train_data,
                        structures,
                        epochs=epoch,
                        batch_size=batch_size,
                        lr=learning_rate,
                        ntarget=len(train_data.columns),
                        n1=n1,
                        n2=n2,
                        n3=n3,
                        save_model=False,
                        
                    )
                    
                    idf = prefix_name + \
                        f"bs{batch_size}_e{epoch}lr{learning_rate}_n1n2n3_{n1}x{n2}x{n3}"
                    model.save_weights(savedir+idf+"_weights.h5")
                    dump(scaler, open(savedir+idf+'_scaler.pkl', 'wb'))
                    ## get score dict
                    results_dict={'batch_size': batch_size,
                                    'epoch': epoch,
                                    'learning_rate' : learning_rate,
                                    'n1': n1,
                                    'n2': n2,
                                    'n3': n3,
                                    }
                    score_dict=megnet_train_val_scores(model,scaler,structures,train_data,
                                            structures_test,test_data,id=idf,savedir=savedir)
                    # Append the current run's results to the dataframe
                    results_dict.update(score_dict)               
                    results_df = results_df.append(results_dict, ignore_index=True)
                    results_df.to_csv("results_tmp.csv", index=False)
    results_df.to_csv("results.csv", index=False)


__all__ = ['hyper_tuning_megnet_model']
