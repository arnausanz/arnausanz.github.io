import pickle

import torch

from model.data_prep import update_data
from model.model import get_split_data, load_model

step_forward = 30
# update_data()

X_train, X_test, y_train, y_test, scalers = get_split_data('xLSTM', 180, num_subwindows=6, steps_fwd=step_forward)

m7 = load_model('model_25')
# Load model 7 scalers
with open(m7.model_config.model_src + '/scalers.pkl', 'rb') as f:
    scalers_7 = pickle.load(f)

pred_7 = scalers_7[-1].inverse_transform(m7.forward(X_test).cpu().detach().numpy())[-1]

print(pred_7)
del m7
torch.mps.empty_cache()


X_train, X_test, y_train, y_test, scalers = get_split_data('LSTM', 270, steps_fwd=step_forward)

m2 = load_model('model_5')
# Load model 2 scalers
with open(m2.model_config.model_src + '/scalers.pkl', 'rb') as f:
    scalers_2 = pickle.load(f)
pred_2 = scalers_2[-1].inverse_transform(m2.forward(X_test).cpu().detach().numpy())[-1]
del m2
torch.mps.empty_cache()
print(pred_2)
