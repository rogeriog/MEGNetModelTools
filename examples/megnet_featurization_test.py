from modnet.preprocessing import MODData
data=MODData.load('../../DATAFILES/matbench_perovskites_moddata.pkl.gz')
import pickle
structures=data.df_structure['structure']
slices=list(range(0,len(structures),1000))+[None]
for idx in range(len(slices)-1):
    #if idx < 4 : 
    #    continue
    print(f"Processing slice {idx+1} out of {len(slices)}")
    MEGNetFeats_struct=get_MEGNetBaseFeatures(structures[slices[idx]:slices[idx+1]])
    pickle.dump(MEGNetFeats_struct,open(f"MEGNetFeats_struct_slice{idx}.pkl", "wb"))
    del MEGNetFeats_struct ## free memory