import torch
import torch.nn as nn

# Read model from /Users/arnausanz/Library/Mobile Documents/com~apple~CloudDocs/Documents/Estudis/UOC/Master Data Science/Semestre 7/TFM/TFM_arnausanz/model/final_models/model_1/model/model.pth
model = nn.Module()
model.load_state_dict(torch.load('/Users/arnausanz/Library/Mobile Documents/com~apple~CloudDocs/Documents/Estudis/UOC/Master Data Science/Semestre 7/TFM/TFM_arnausanz/model/final_models/model_1/model/model.pth'))

print(model)