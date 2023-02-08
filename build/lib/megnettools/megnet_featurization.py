from megnet.utils.models import load_model, AVAILABLE_MODELS
import numpy as np
from keras.models import Model
import warnings
import pandas as pd
from .megnet_setup_evaluate import load_model_scaler, megnet_evaluate_structures
warnings.filterwarnings("ignore")
from typing import Tuple, Any
# print(AVAILABLE_MODELS)
def get_MEGNetBaseFeatures(structures):
    MEGNetFeats_structs=[]
    for model_name in ['Eform_MP_2019','Efermi_MP_2019','Bandgap_MP_2018','logK_MP_2019','logG_MP_2019']:
        model=load_model(model_name) 
        intermediate_layer_model = Model(inputs=model.input,
                             outputs=model.layers[-3].output)   
        MEGNetModel_structs=[]
        for s in structures:
            try:
                graph = model.graph_converter.convert(s)
                inp = model.graph_converter.graph_to_input(graph)
                pred = intermediate_layer_model.predict(inp, verbose=False)
                model_struct=pd.DataFrame([pred[0][0]], 
                                          columns=[f"{model_name}_{idx+1}" for idx in 
                                                   range(len(pred[0][0]))])
                MEGNetModel_structs.append(model_struct)
            except Exception as e:
                print(e)
                print("Probably an invalid structure was passed to the model, continuing..")
                model_struct=pd.DataFrame([np.nan]*32, 
                                          columns=[f"{model_name}_{idx+1}" for idx in 
                                                   range(len(pred[0][0]))])
                continue
        ## now append the columns with the layer of each model
        MEGNetModel_structs=pd.concat(MEGNetModel_structs,axis=0)
        MEGNetFeats_structs.append(MEGNetModel_structs)
        print(f"Features calculated for model {model_name}.")
    ## now every structure calculated with each model is combined in a final dataframe
    MEGNetFeats_structs=pd.concat(MEGNetFeats_structs,axis=1)
    return MEGNetFeats_structs


def get_MEGNetFeatures(structures,
                       n_targets : int = 1,
                       neuron_layers : Tuple[int] = (64,32,16), 
                       model=None, 
                       model_file=None, 
                       scaler=None,
                       scaler_file=None,
                       **kwargs):
    '''From a specified model, either passed directly or loaded from file
    scaler is optional to scale back the produced output. 
    Reads a set of structures filters them'''
    model_name=kwargs.get('model_name','myMEGNetModel')
    if model is None:
        model,scaler=load_model_scaler(n_targets=n_targets, 
                        neuron_layers=neuron_layers,
                        model_file=model_file, scaler_file=scaler_file, 
                        **kwargs)

    MEGNetFeatsDF=[]    
    structures_valid,ypred=megnet_evaluate_structures(model,structures)
    print(ypred)
    for s in structures:
        if s in list(structures_valid):
            s_idx = list(structures_valid).index(s)
            p = ypred[s_idx]
            if scaler is None:
                feat_data=pd.DataFrame([p],columns=[f"MEGNet_{model_name}_{idx+1}" for idx in range(n_targets)])
                struct=pd.DataFrame({'structure': [s]})
                modeldata_struct = pd.concat([struct,feat_data], axis=1)
            else:
                feat_data=pd.DataFrame(scaler.inverse_transform(p.reshape(1, -1)),
                                    columns=[f"MEGNet_{model_name}_{idx+1}" for idx in range(n_targets)])
                struct=pd.DataFrame({'structure': [s]})
                modeldata_struct = pd.concat([struct,feat_data], axis=1)
        else:
            feat_data=pd.DataFrame([[np.nan] * n_targets],columns=[f"MEGNet_{model_name}_{idx+1}" for idx in range(n_targets)])
            struct=pd.DataFrame({'structure': [s]})
            modeldata_struct = pd.concat([struct,feat_data], axis=1)        
        MEGNetFeatsDF.append(modeldata_struct)
    MEGNetFeatsDF = pd.concat(MEGNetFeatsDF,axis=0)  
    MEGNetFeatsDF = MEGNetFeatsDF.reset_index(drop=True)
    return MEGNetFeatsDF
    

__all__ = ['get_MEGNetFeatures', 'get_MEGNetBaseFeatures']