import pandas as pd

def text_to_df(path):
    with open(path) as f:
        lines=f.readlines()

    data_lines = []
    for line in lines:
        if line.strip().startswith("Model with highest F1"):
            break
        if line.strip():
            line=line.strip()
            clean_line=" ".join(line.split())
            data_lines.append(clean_line.split())

    df_metrics=pd.DataFrame(data_lines[1:],columns=data_lines[0])
    df_metrics=df_metrics.astype("float")
    return df_metrics