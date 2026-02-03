import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# --- BAYESIAN COMPONENTS ---
class BayesLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_var=36.0):
        super().__init__()
        self.register_buffer('prior_var', torch.tensor(prior_var))
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.01))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).fill_(-6))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.01))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).fill_(-6))

    def forward(self, x):
        w_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight = self.weight_mu + w_sigma * torch.randn_like(w_sigma)
        b_sigma = torch.log1p(torch.exp(self.bias_rho))
        bias = self.bias_mu + b_sigma * torch.randn_like(b_sigma)
        return torch.matmul(x, weight.t()) + bias

class BayesLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = BayesLinear(input_dim + hidden_dim, 4 * hidden_dim)
        self.output = BayesLinear(hidden_dim, output_dim)
        
    def forward(self, x):
        batch_size, seq_size, _ = x.shape
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        for t in range(seq_size):
            combined = torch.cat((x[:, t, :], h_t), 1)
            i, f, c_hat, o = torch.split(self.gates(combined), self.hidden_dim, dim=1)
            c_t = torch.sigmoid(f) * c_t + torch.sigmoid(i) * torch.tanh(c_hat)
            h_t = torch.sigmoid(o) * torch.tanh(c_t)
        return self.output(h_t)

# --- DATA PREPARATION ---
df_raw = pd.read_csv('ibtracs_BB_only_edited6.csv')

df_raw['ISO_TIME'] = pd.to_datetime(df_raw['ISO_TIME'])
df_raw['month'] = df_raw['ISO_TIME'].dt.month
df_raw['day'] = df_raw['ISO_TIME'].dt.day
df_raw['time'] = df_raw['ISO_TIME'].dt.hour

feature_weights = {
    'SEASON': 0.95, 'NUMBER': 0.90, 'NAME': 0.80, 'NATURE': 0.70,
    'TRACK_TYPE': 0.40, 'DIST2LAND': 0.30, 'LANDFALL': 0.20,
    'USA_POCI': 0.10, 'USA_ROCI': 0.05, 'USA_RMW': 0.00,
    'NEWDELHI_WIND': -0.10, 'NEWDELHI_PRES': -0.15, 'STORM_SPEED': -0.20,
    'STORM_DIR': -0.25, 'month': -0.30, 'day': -0.35, 'time': -0.40
}
target_cols = ['LAT', 'LON']
active_features = [col for col in feature_weights.keys() if col in df_raw.columns]

le = LabelEncoder()
for col in ['NAME', 'NATURE', 'TRACK_TYPE']:
    if col in df_raw.columns:
        df_raw[col] = le.fit_transform(df_raw[col].astype(str))

df_proc = df_raw.copy()
for col in active_features:
    df_proc[col] = df_proc[col].fillna(0) * feature_weights[col]

f_scaler, t_scaler = MinMaxScaler(), MinMaxScaler()
input_cols = active_features + target_cols
df_proc[input_cols] = f_scaler.fit_transform(df_proc[input_cols].fillna(0))
t_scaler.fit(df_raw[target_cols].fillna(0))

win = 8 
X_list, y_list, sid_list = [], [], []
for sid, group in df_proc.groupby('SID'):
    if len(group) > win:
        data = group[input_cols].values
        raw_targets = df_raw.loc[group.index, target_cols].fillna(0)
        scaled_targets = t_scaler.transform(raw_targets)
        for i in range(len(data) - win):
            X_list.append(data[i:i+win])
            y_list.append(scaled_targets[i+win])
            sid_list.append(sid)

X = torch.tensor(np.array(X_list)).float()
y = torch.tensor(np.array(y_list)).float()
sid_arr = np.array(sid_list)
split = int(0.8 * len(X))

train_loader = DataLoader(TensorDataset(X[:split], y[:split]), batch_size=256, shuffle=True)

# --- TRAINING ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BayesLSTM(X.shape[2], 128, 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

print("Starting Training...")
for epoch in tqdm(range(100), desc="Training"):
    model.train()
    epoch_loss = 0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(train_loader):.6f}")

# --- EVALUATION ---
model.train() # Enable stochastic sampling
print("\nEvaluating Model...")
with torch.no_grad():
    # Helper to get RMSE for a dataset
    def get_rmse(X_data, y_data):
        # 50 samples per prediction for stochastic mean
        preds = torch.stack([model(X_data.to(device)) for _ in range(50)]).mean(0).cpu().numpy()
        real = t_scaler.inverse_transform(pd.DataFrame(y_data.numpy(), columns=target_cols))
        pred_real = t_scaler.inverse_transform(pd.DataFrame(preds, columns=target_cols))
        return np.sqrt(np.mean((pred_real - real)**2)), pred_real, real

    train_rmse, _, _ = get_rmse(X[:split], y[:split])
    test_rmse, p_real, y_real = get_rmse(X[split:], y[split:])

    print("-" * 30)
    print(f"Train RMSE: {train_rmse:.4f} degrees")
    print(f"Test RMSE:  {test_rmse:.4f} degrees")
    print("-" * 30)

# --- CARTOPY TRAJECTORY VALIDATION ---
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE)
ax.set_extent([75, 100, 5, 27], crs=ccrs.PlateCarree())

test_sids = sid_arr[split:]
unique_sids = np.unique(test_sids)[:5]

for i, sid in enumerate(unique_sids):
    idx = np.where(test_sids == sid)[0]
    l_act = 'Actual' if i == 0 else None
    l_pre = 'Pred' if i == 0 else None
    
    ax.plot(y_real[idx, 1], y_real[idx, 0], 'k-o', markersize=3, label=l_act, transform=ccrs.PlateCarree())
    ax.plot(p_real[idx, 1], p_real[idx, 0], 'r--x', markersize=4, label=l_pre, transform=ccrs.PlateCarree())

plt.title(f"Stochastic LSTM\nTrain RMSE: {train_rmse:.4f}°, Test RMSE: {test_rmse:.4f}°")
plt.legend(loc='lower right')
plt.show()