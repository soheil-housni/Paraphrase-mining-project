import pandas as pd

def hp_extraction(path):
    with open(path) as f:
        lines=f.readlines()

    data_lines=[]
    for i in range(len(lines)):
        if lines[i].strip().startswith("Model with highest F1 score:"):
            line1=lines[i+1].strip().split()
            line2=lines[i+2].strip().split()
            data_lines.append(line1)
            data_lines.append(line2)
            break

    data_lines[1]=[data_lines[1]]
    df=pd.DataFrame(data=data_lines[1],columns=data_lines[0])
    df=df.astype(float)
    df["use_n_layers"]=df["use_n_layers"].astype(int)

    if "use_n_layers_cross_att" in df.columns:
        df["use_n_layers_cross_att"]=df["use_n_layers_cross_att"].astype(int)
    
    return df