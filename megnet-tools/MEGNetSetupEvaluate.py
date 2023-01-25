import numpy as np
from typing import Type, List
from megnet.models import MEGNetModel
from megnet.data.crystal import CrystalGraph
import pickle
from pickle import dump, load
import copy
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
from figrecipes import PlotlyFig
from sklearn.metrics import mean_absolute_error
from typing import Tuple, Any

def model_setup(ntarget: int=None,
                n1: int=64,
                n2: int=32,
                n3: int=16,
                nfeat_bond: int=100,
                **kwargs) -> Any:
    """
    This function takes in a number of optional parameters for creating a MEGNet model, such as number of neurons 
    in different layers, and the number of features for bonds.
    It returns an instance of a MEGNet model which is set up with the given parameters.
    """
    nfeat_bond = nfeat_bond
    r_cutoff = kwargs.get('r_cutoff',5)
    gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
    gaussian_width = kwargs.get('gaussian_width',0.5)
    graph_converter = CrystalGraph(cutoff=r_cutoff)

    model = MEGNetModel(graph_converter=graph_converter, centers=gaussian_centers, width=gaussian_width,
                        ntarget=ntarget, n1=n1, n2=n2, n3=n3)
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    print(short_model_summary.splitlines()[-4])
    return model

def regression_model(model: object,  scaler: object,
                    structures: List[str], targets: pd.DataFrame, 
                    detailed_output: bool = False, 
                    **kwargs) -> None:
    ''' This function takes in the following required arguments:

    model : the model that is used to predict the structures
    structures : the list of structures to predict
    targets : the list of actual values to compare the predictions to
    scaler : the scaler used to scale the targets
    
    It also takes in an optional argument:
    detailed_output : a boolean that controls whether or not to output detailed plots (default is False)
    
    It also takes additional keyword arguments (**kwargs) for example:
    id : a string that represents the id of the model
    labels : a list of labels for the structures
    savedir : the directory to save the output files
    It uses the kwargs.get() method to get the value of the 'id' and 'labels' keys, if they don't exist, it assigns them empty string and empty list respectively.

    The function then separates the valid and invalid structures according to MEGNet model and calculate the mean absolute error (MAE) between 
    the predictions and the actual values. It then saves the MAE values to a text file and if detailed_output is true it will create some 
    plots using the library figrecipes and save the plots in a directory 'figs'

    The function returns None, it only prints the MAE values on the screen and saves the results in a file.
    Predicts using the model and then compares to actual values in plotly figure
    '''
    id=kwargs.get('id','')
    labels=kwargs.get('labels',['']*len(structures))
    labels=np.array(labels)
    
    # have to exclude structures that dont form compatible graphs and their corresponding targets.
    structures_valid = []
    targets_valid = []
    labels_valid = []
    structures_invalid = []
    for s, p, l in zip(structures, targets.values, labels):
       try:
           graph = model.graph_converter.convert(s)
           structures_valid.append(s)
           targets_valid.append(np.nan_to_num(scaler.transform(p.reshape(1,-1))))
           labels_valid.append(l)
       except:
           structures_invalid.append(s)
    structures_valid=np.array(structures_valid)

    y=np.array(targets_valid)
    labels=np.array(labels_valid)
    print(f"Following invalid structures: {structures_invalid}.")
    y_pred=model.predict_structures(structures_valid)
    print(y_pred)
    # y_pred=y_pred.flatten()
    y=y.squeeze()
    MAEs=mean_absolute_error(y, y_pred, multioutput='raw_values')

    maes_text=f'MAEs mean: {MAEs.mean()}'
    savedir=kwargs.get('savedir','./')
    if not os.path.exists(path):
        os.mkdir(path)
    os.chdir(path)
    with open(savedir+'MAE_'+id+'.txt', 'w') as f: f.write(maes_text)
    print(maes_text)

    #MAEs=pd.DataFrame(MAEs.reshape(1,-1),columns=targets.columns)
    MAEs=pd.DataFrame(MAEs,index=targets.columns)
    if detailed_output:
        # Create a new directory because it does not exist
        path='./figs'
        if not os.path.exists(path):
            os.mkdir(path)
        os.chdir(path)
        histo_name="HISTO_MAE"+str(id)
        pf_hist = PlotlyFig(x_title=kwargs.get('x_title','Expected Y'),
                          y_title=kwargs.get('y_title','Predicted Y'),
                          title=kwargs.get('title','Regression of model'),
                          mode='offline',
                         filename=histo_name+"_top10.html" )
        MAEs_top=MAEs.sort_values(by=0, ascending=False)

        #MAEs=MAEs[MAEs.columns[0:10]]
        MAEs_top=MAEs_top[:10]
        pf_hist.bar(data=MAEs_top, cols=MAEs_top.index)
        
        pf_hist = PlotlyFig(x_title=kwargs.get('x_title','Expected Y'),
                          y_title=kwargs.get('y_title','Predicted Y'),
                          title=kwargs.get('title','Regression of model'),
                          mode='offline',
                          filename=histo_name+"_bottom10.html" )
        MAEs_bot=MAEs.sort_values(by=0, ascending=True)
        #MAEs=MAEs[MAEs.columns[0:10]]
        MAEs_bot=MAEs_bot[:10]
        pf_hist.bar(data=MAEs_bot, cols=MAEs_bot.index)

        for idx, feat in enumerate(targets.columns):
        #from sklearn.model_selection import cross_val_predict
            pf_target = PlotlyFig(x_title=kwargs.get('x_title','Expected Y'),
                              y_title=kwargs.get('y_title','Predicted Y'),
                              title=kwargs.get('title','Regression of model'),
                              mode='offline',
                              filename=feat+"_"+id+"_pred_vs_real.html" )
            if y.ndim == 1:
                y_feat=y
                y_feat_pred=y_pred
            else:
                y_feat=y[:,idx]
                y_feat_pred=y_pred[:,idx]
            pf_target.xy(xy_pairs=[(y_feat, y_feat_pred), ([min(y_feat), max(y_feat)], [min(y_feat), max(y_feat)])],
              labels=labels, modes=['markers', 'lines'],
              lines=[{}, {'color': 'black', 'dash': 'dash'}], showlegends=False)

        os.chdir('../')
    return MAEs.mean().to_numpy()[0]

def get_scaler(targets):
    scaler=MinMaxScaler()
    scaler.fit(targets.values)
    return scaler


def generate_model_scaler(df_featurized_train: pd.DataFrame,
    df_structure_train: pd.DataFrame,
    ntarget: int = None, save_model=True, id: str = '',
    **kwargs) -> Tuple[Any, Any]:
    """
    This function takes in a dataframe of featurized training data and a dataframe of structure training data, 
    along with other optional parameters.
    It returns a tuple of a trained model and the corresponding scaler.
    """
    model=model_setup(ntarget=ntarget, n1=kwargs.get('n1',64), n2=kwargs.get('n2',32), n3=kwargs.get('n3',16))
    # Model training
    # Here, `structures` is a list of pymatgen Structure objects.
    # `targets` is a corresponding list of properties.
    structures=df_structure_train
    targets=df_featurized_train
    ## the following will scale and process the targets as well as filter valid structures
    graphs_valid = []
    targets_valid = []
    structures_invalid = []
    if kwargs.get('prev_scaler',False):
        scaler= load(open(kwargs.get('prev_scaler'), 'rb'))
    else:
        scaler=get_scaler(targets)

    for s, p in zip(structures, targets.values):
        try:
            graph = model.graph_converter.convert(s)
            graphs_valid.append(graph)
            ## Standardize data and substitute nan to 0, that is, the mean.
            targets_valid.append(np.nan_to_num(scaler.transform(p.reshape(1,-1))))
        except:
            structures_invalid.append(s)

    # train the model using valid graphs and targets
    model.train_from_graphs(graphs_valid, targets_valid, 
                            batch_size=kwargs.get('batch_size',64), 
                            epochs=kwargs.get('epochs',100),
                            prev_model=kwargs.get('prev_model',None)) ##prev_model uses loads_weights 
    if save_model:
        model.save_weights(f"MEGNetModel{id}_weights.h5")
        dump(scaler, open(f'MEGNetModel{id}_scaler.pkl', 'wb'))
    return (model,scaler)

def load_model_scaler(id: str, 
                      n_targets: int, mode: str = 'partial', 
                      n1: int = 64, n2: int = 32, n3: int = 16, 
                      **kwargs) -> Tuple[Any, Any]:
    """
    This function takes in an id, number of targets, a mode, and other optional parameters for loading a previously trained MEGNet model and its corresponding scaler.
    It returns a tuple of the loaded model and scaler.
    """
    model=model_setup(ntarget=n_targets, n1=n1, n2=n2, n3=n3 )
    namemodel=kwargs.get("modeldir","./models/")+"ModelMEGNet_"+mode+"_"+id
    model.load_weights(f"MEGNetModel{id}_weights.h5")
    MEGNetModel=model  
    scaler = load(open(f'MEGNetModel{id}_scaler.pkl', 'rb'))
    return (MEGNetModel, scaler)


