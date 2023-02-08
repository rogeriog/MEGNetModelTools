import numpy as np
from typing import List
from megnet.models import MEGNetModel
from megnet.data.crystal import CrystalGraph
from pickle import dump, load
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
from figrecipes import PlotlyFig
from typing import Tuple, Any
from scipy.spatial.distance import correlation, cosine
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def model_setup(ntarget: int = None,
                **kwargs) -> Any:
    """
    This function takes in a number of optional parameters for creating a MEGNet model, such as number of neurons 
    in different layers, and the number of features for bonds.
    It returns an instance of a MEGNet model which is set up with the given parameters.
    """
    ## default architecture:
    n1=kwargs.get('n1', 64) 
    n2=kwargs.get('n2', 32) 
    n3=kwargs.get('n3', 16)
    nfeat_bond = kwargs.get('nfeat_bond', 100)
    r_cutoff = kwargs.get('r_cutoff', 5)
    gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
    gaussian_width = kwargs.get('gaussian_width', 0.5)
    graph_converter = CrystalGraph(cutoff=r_cutoff)

    model = MEGNetModel(graph_converter=graph_converter, centers=gaussian_centers, width=gaussian_width,
                        ntarget=ntarget, **kwargs)
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    print(short_model_summary.splitlines()[-4])
    return model

def megnet_evaluate_structures(model, structures,
                               targets=None,
                               scaler=None, **kwargs):

    labels = kwargs.get('labels', ['']*len(structures))

    noTargets=False
    if targets is None:
        target_values = np.array([1]*len(structures))
        noTargets=True
    else:
        if isinstance(targets, pd.DataFrame):
            target_values=targets.values
        else:
            target_values=targets
    # have to exclude structures that dont form compatible graphs and their corresponding targets.
    structures_valid = []
    targets_valid = []
    labels_valid = []
    structures_invalid = []
    for s, p, l in zip(structures, target_values, labels):
        try:
            graph = model.graph_converter.convert(s)
            structures_valid.append(s)
            if scaler is not None:
                targets_valid.append(np.nan_to_num(
                    scaler.transform(p.reshape(1, -1))))
            else:
                targets_valid.append(p)
            labels_valid.append(l)
        except:
            structures_invalid.append(s)
    # structures_valid = np.array(structures_valid)

    y = np.array(targets_valid)
    y = y.squeeze()
    labels = np.array(labels_valid)
    print(f"Following invalid structures: {structures_invalid}.")
    print(type(structures_valid),structures_valid)
    ypred = model.predict_structures(list(structures_valid))
    if noTargets:
        return (structures_valid,ypred)
    if not noTargets:
        return (structures_valid,ypred, y, labels)
    # y_pred=y_pred.flatten()
    

def megnet_regression_model(model: object,  scaler: object,
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
    id = kwargs.get('id', '')

    structures_valid, ypred, y, labels = megnet_evaluate_structures(model, 
                                                                    structures, 
                                                                    targets=targets, 
                                                                    scaler=scaler)
    # labels = kwargs.get('labels', ['']*len(structures))
    # labels = np.array(labels)

    # # have to exclude structures that dont form compatible graphs and their corresponding targets.
    # structures_valid = []
    # targets_valid = []
    # labels_valid = []
    # structures_invalid = []
    # for s, p, l in zip(structures, targets.values, labels):
    #     try:
    #         graph = model.graph_converter.convert(s)
    #         structures_valid.append(s)
    #         targets_valid.append(np.nan_to_num(
    #             scaler.transform(p.reshape(1, -1))))
    #         labels_valid.append(l)
    #     except:
    #         structures_invalid.append(s)
    # structures_valid = np.array(structures_valid)

    # y = np.array(targets_valid)
    # labels = np.array(labels_valid)
    # print(f"Following invalid structures: {structures_invalid}.")
    # ypred = model.predict_structures(structures_valid)
    # print(ypred)
    # # y_pred=y_pred.flatten()
    # y = y.squeeze()
    MAEs = mean_absolute_error(y, ypred, multioutput='raw_values')

    maes_text = f'MAEs mean: {MAEs.mean()}'
    savedir = kwargs.get('savedir', './')
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    with open(savedir+'MAE_'+id+'.txt', 'w') as f:
        f.write(maes_text)
    print(maes_text)

    # MAEs=pd.DataFrame(MAEs.reshape(1,-1),columns=targets.columns)
    MAEs = pd.DataFrame(MAEs, index=targets.columns)
    if detailed_output:
        # Create a new directory because it does not exist
        path = './figs'
        if not os.path.exists(path):
            os.mkdir(path)
        os.chdir(path)
        histo_name = "HISTO_MAE"+str(id)
        pf_hist = PlotlyFig(x_title=kwargs.get('x_title', 'Expected Y'),
                            y_title=kwargs.get('y_title', 'Predicted Y'),
                            title=kwargs.get('title', 'Regression of model'),
                            mode='offline',
                            filename=histo_name+"_top10.html")
        MAEs_top = MAEs.sort_values(by=0, ascending=False)

        # MAEs=MAEs[MAEs.columns[0:10]]
        MAEs_top = MAEs_top[:10]
        pf_hist.bar(data=MAEs_top, cols=MAEs_top.index)

        pf_hist = PlotlyFig(x_title=kwargs.get('x_title', 'Expected Y'),
                            y_title=kwargs.get('y_title', 'Predicted Y'),
                            title=kwargs.get('title', 'Regression of model'),
                            mode='offline',
                            filename=histo_name+"_bottom10.html")
        MAEs_bot = MAEs.sort_values(by=0, ascending=True)
        # MAEs=MAEs[MAEs.columns[0:10]]
        MAEs_bot = MAEs_bot[:10]
        pf_hist.bar(data=MAEs_bot, cols=MAEs_bot.index)

        for idx, feat in enumerate(targets.columns):
            
            pf_target = PlotlyFig(x_title=kwargs.get('x_title', 'Expected Y'),
                                  y_title=kwargs.get('y_title', 'Predicted Y'),
                                  title=kwargs.get(
                                      'title', 'Regression of model'),
                                  mode='offline',
                                  filename=feat+"_"+id+"_pred_vs_real.html")
            if y.ndim == 1:
                y_feat = y
                y_feat_pred = ypred
            else:
                y_feat = y[:, idx]
                y_feat_pred = ypred[:, idx]
            pf_target.xy(xy_pairs=[(y_feat, y_feat_pred), ([min(y_feat), max(y_feat)], [min(y_feat), max(y_feat)])],
                         labels=labels, modes=['markers', 'lines'],
                         lines=[{}, {'color': 'black', 'dash': 'dash'}], showlegends=False)

        os.chdir('../')
    resultsmodel={'MAE':MAEs.mean().to_numpy()[0]}
    ## now other relevant scores
    for metric in [correlation, cosine]:
        metric_result=np.array([metric(y[i],ypred[i]) for i in range(len(y))]).mean()
        print(metric.__name__,metric_result)
        resultsmodel[metric.__name__] = metric_result
    
    rmse_result=np.sqrt(mean_squared_error(y, ypred))
    print('RMSE:',rmse_result)
    resultsmodel['RMSE'] = rmse_result
    
    r2_result=r2_score(y, ypred, multioutput='variance_weighted')
    print('r2:',r2_result)
    resultsmodel['R2'] = r2_result
    
    yzeros=np.zeros(y.shape)
    results_rmse0=np.sqrt(mean_squared_error(y, yzeros))
    print('RMSE zero-vector:',results_rmse0)
    resultsmodel['RMSE zero-vector'] = results_rmse0
    
    return resultsmodel

def megnet_train_val_scores(model,scaler,X_train,y_train,X_test,y_test,id='MEGNetModel',savedir='./'):
    # now get MAEs
    scores_dict = megnet_regression_model(
        model, scaler, X_test, y_test,  id=id+'_val', savedir=savedir)
    # getting MAE train
    train_scores_dict = megnet_regression_model(
        model,  scaler, X_train, y_train,id=id+'_train', savedir=savedir)    
    train_scores_dict = {f"{k}_train": v for k, v in train_scores_dict.items()}
    scores_dict.update(train_scores_dict)
    return scores_dict

def get_scaler(targets):
    scaler = MinMaxScaler()
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
    model = model_setup(ntarget=ntarget, **kwargs)
    # Model training
    # Here, `structures` is a list of pymatgen Structure objects.
    # `targets` is a corresponding list of properties.
    structures = df_structure_train
    targets = df_featurized_train
    # the following will scale and process the targets as well as filter valid structures
    graphs_valid = []
    targets_valid = []
    structures_invalid = []
    if kwargs.get('prev_scaler', False):
        scaler = load(open(kwargs.get('prev_scaler'), 'rb'))
    else:
        scaler = get_scaler(targets)

    for s, p in zip(structures, targets.values):
        try:
            graph = model.graph_converter.convert(s)
            graphs_valid.append(graph)
            # Standardize data and substitute nan to 0, that is, the mean.
            targets_valid.append(np.nan_to_num(
                scaler.transform(p.reshape(1, -1))))
        except:
            structures_invalid.append(s)

    # train the model using valid graphs and targets
    model.train_from_graphs(graphs_valid, targets_valid,
                            batch_size=kwargs.get('batch_size', 64),
                            epochs=kwargs.get('epochs', 100),
                            prev_model=kwargs.get('prev_model', None))  # prev_model uses loads_weights
    if save_model:
        model.save_weights(f"MEGNetModel{id}_weights.h5")
        dump(scaler, open(f'MEGNetModel{id}_scaler.pkl', 'wb'))
    return (model, scaler)


def load_model_scaler(id: str = '',
                      n_targets: int = 1 ,
                      neuron_layers: Tuple[int] = (64,32,16),
                      **kwargs) -> Tuple[Any, Any]:
    """
    This function takes in an id, number of targets, a mode, and other optional parameters for loading a previously trained MEGNet model and its corresponding scaler.
    It returns a tuple of the loaded model and scaler.
    """
    n1,n2,n3=neuron_layers
    model = model_setup(ntarget=n_targets, n1=n1, n2=n2, n3=n3,
                        **kwargs)
    modelpath_id = kwargs.get("modeldir", "./")+id
    model_file=kwargs.get('model_file',f"{modelpath_id}_weights.h5")
    scaler_file=kwargs.get('scaler_file',f'{modelpath_id}_scaler.pkl')
    model.load_weights(model_file)
    try: ## if scaler not found, it will be None
        scaler = load(open(scaler_file, 'rb'))
    except:
        scaler = None
    return (model, scaler)


__all__ = ['model_setup', 'megnet_regression_model', 'get_scaler',
           'generate_model_scaler', 'load_model_scaler']
