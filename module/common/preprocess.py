import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from typing import Tuple


def preprocess_train(data_dir: str) ->  Tuple[np.ndarray, np.ndarray, dict]:
    """
    Preprocess train data

    Args:
        data_dir (str): parent directory that contains train_input, triain_target

    Returns:
        Tuple[np.ndarray, np.ndarray, dict]: X, Y, sample num
    """
    train_input_dir = f"{data_dir}/train_input"
    train_target_dir = f"{data_dir}/train_target"
    input_lst = os.listdir(train_input_dir)
    target_lst = os.listdir(train_target_dir)
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
