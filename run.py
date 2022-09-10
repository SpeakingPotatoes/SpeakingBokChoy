#%%
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

#%%
train_input_dir = "/Users/wonhyung64/data/bok_choy/train_input"
test_input_dir = "/Users/wonhyung64/data/bok_choy/test_input"
train_target_dir = "/Users/wonhyung64/data/bok_choy/train_target"
test_target_dir = "/Users/wonhyung64/data/bok_choy/test_target"

#%%
data_dir = "/Users/wonhyung64/data/bok_choy"

def preprocess(data_dir: str) ->  Tuple[np.ndarray, np.ndarray, dict]:
    """
    Preprocess train data

    Args:
        data_dir (str): parent directory that contains train_input, triain_target

    Returns:
        Tuple[np.ndarray, np.ndarray, dict]: X, Y, sample num
    """
    input_lst = os.listdir(f"{data_dir}/train_input_dir")
    target_lst = os.listdir(f"{data_dir}/train_target_dir")
    input_lst.sort()
    target_lst.sort()

    X, Y = [], []
    progress = tqdm(range(len(input_lst)))
    for i in progress:
        input = pd.read_csv(f"{train_input_dir}/{input_lst[i]}")
        target = pd.read_csv(f"{train_target_dir}/{target_lst[i]}")
        input["시간"] = input["시간"].map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        input["시간"] = input["시간"].map(lambda x: datetime(x.year, x.month, x.day))
        target["시간"] = target["시간"].map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        target["시간"] = target["시간"].map(lambda x: datetime(x.year, x.month, x.day))
        date_lst = input["시간"].tolist()
        date_lst = list(set(date_lst))
        date_lst.sort()
        for date in date_lst:
            input_date = datetime(date.year, date.month, date.day)
            target_date = input_date + timedelta(days=1)
            input_df = input[input["시간"] == input_date]
            len(input_df.columns.tolist())
            target_df = target[target["시간"] == target_date]
            if input_df.isnull().any().any() or len(target_df) == 0:
                continue
            X.append(input_df.iloc[:,1:].mean().to_numpy())
            Y += target_df["rate"].tolist()

    X, Y = np.array(X), np.array(Y)
    response = {
        "X_shape": X.shape,
        "Y_shape": Y.shape
    }

    return X, Y, response
# %%

# %%
input_lst = os.listdir(train_input_dir)
target_lst = os.listdir(train_target_dir)
input_lst.sort()
target_lst.sort()

cols = []
progress = tqdm(range(len(input_lst)))
for i in progress:
    
    input = pd.read_csv(f"{train_input_dir}/{input_lst[i]}")
    df_cols = input.columns.tolist()
    if i > 0:
        if tmp_cols != df_cols: break
    # len(df_cols)
    if len(df_cols) != 38: break
    # if "펌프작동남은시간.1" in input.columns.tolist(): break
    cols += df_cols
    tmp_cols = df_cols.copy()

input_1 = pd.read_csv(f"{train_input_dir}/{input_lst[i-1]}")
input.iloc[:, 8:12]
input_1.iloc[:, 8:12]
print(tmp_cols)
print(df_cols)

tmp = set(cols)
len(tmp)

'''
일간누적분무량
펌프작동남은시간.1
외부온도관측치 추정
외부습도관측지 추정
'''