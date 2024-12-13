import pickle
import pandas as pd
import torch

from model.data_prep import update_data
from model.model import get_split_data, load_model

step_forward = 30
# update_data()

X_train, X_test, y_train, y_test, scalers = get_split_data('LSTM', 270, steps_fwd=step_forward)

m2 = load_model('model_5')
# Load model 2 scalers
with open(m2.model_config.model_src + '/scalers.pkl', 'rb') as f:
    scalers_2 = pickle.load(f)
pred_2 = scalers_2[-1].inverse_transform(m2.forward(X_test).cpu().detach().numpy())[-1]
del m2
torch.mps.empty_cache()

sensors = pd.read_csv('model/final_data/sensor_codes.csv')

preds_df = (pd.DataFrame(pred_2, index = sensors['name'], columns=['value']))
preds_df['value'] = preds_df['value'].apply(lambda x: str(round(x, 2))+'hm3')
preds_df.to_csv('web_source.csv')
