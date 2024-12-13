from model.data_prep import update_data
from model.model import get_split_data, load_model

step_forward = 30
update_data()
X_train, X_test, y_train, y_test, scalers = get_split_data('LSTM', 270, steps_fwd=step_forward)

m = load_model('model_5')

print(m.forward(X_test[-2:])[-1])
