#%%
from module.common.preprocess import preprocess_train


#%%
data_dir = "/Users/wonhyung64/data/bok_choy"
X, Y, response = preprocess_train(data_dir)

# %%
