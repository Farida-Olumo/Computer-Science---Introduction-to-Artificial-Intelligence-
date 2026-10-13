import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def process_data(data_frame):
    data_frame = data_frame.fillna('NA')
    onehot_data = pd.get_dummies(data_frame, dtype=np.uint8)
    # one class column (high=1, low=0)
    if 'class_low' in onehot_data.columns:
        onehot_data = onehot_data.drop('class_low', axis=1)
    onehot_data = onehot_data.rename(columns={"class_high": "class"})
    return onehot_data

class MyDataset(Dataset):
    def __init__(self, data_frame):
        x = data_frame.drop('class', axis=1).values
        y = data_frame['class'].values
        self.x_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y_train)
    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


class FlexibleNN(nn.Module):
    def __init__(self, input_size, num_layers, act_fn='relu', use_dropout=False):
        super().__init__()
        layers = []
        in_dim = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, 36))
            layers.append(nn.ReLU() if act_fn == 'relu' else nn.Sigmoid())
            if use_dropout:
                layers.append(nn.Dropout(p=0.5))
            in_dim = 36
        layers.append(nn.Linear(36, 2)) 
        self.model = nn.Sequential(*layers)

    def forward(self, x):
       
        return self.model(x)


def train_epoch(data_loader, model, loss_fn, optimizer):
    model.train()
    correct = 0
    for X, y in data_loader:
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    return correct / len(data_loader.dataset)

def eval_epoch(data_loader, model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in data_loader:
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    return correct / len(data_loader.dataset)

def run_experiment(num_layers, act_fn='relu', use_dropout=False):
   
    input_size = train_data.x_train.shape[1]
    model = FlexibleNN(input_size, num_layers, act_fn, use_dropout)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_acc_history = []
    test_acc_history = []

    for epoch in range(100): # 100 epochs per experiment
        t_acc = train_epoch(train_loader, model, loss_fn, optimizer)
        v_acc = eval_epoch(test_loader, model)
        train_acc_history.append(t_acc)
        test_acc_history.append(v_acc)
    return train_acc_history, test_acc_history


df_raw = pd.read_csv("categorical_real_estate.csv", na_values=['NA', 'None'])
df_proc = process_data(df_raw)

train_data = MyDataset(df_proc.iloc[:1000])
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_data = MyDataset(df_proc.iloc[1000:])
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)


print("Running Arch Experiment...")
arch_results = {f"{i} Layers": run_experiment(num_layers=i) for i in [1, 2, 3, 4]}


print("Running Activation Experiment...")
act_results = {
    "ReLU": run_experiment(num_layers=2, act_fn='relu'),
    "Sigmoid": run_experiment(num_layers=2, act_fn='sigmoid')
}


print("Running Dropout Experiment...")
drop_results = {
    "No Dropout": run_experiment(num_layers=4, use_dropout=False),
    "With Dropout": run_experiment(num_layers=4, use_dropout=True)
}


def save_plots(results_dict, title, filename_prefix):
    # Train Plot
    plt.figure()
    for label, (train_h, _) in results_dict.items():
        plt.plot(train_h, label=label)
    plt.title(f"{title} (Training)")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.savefig(f"{filename_prefix}_train.png")
    
    # Test Plot
    plt.figure()
    for label, (_, test_h) in results_dict.items():
        plt.plot(test_h, label=label)
    plt.title(f"{title} (Testing)")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.savefig(f"{filename_prefix}_test.png")

save_plots(arch_results, "Architecture Comparison", "arch")
save_plots(act_results, "Activation Comparison", "act")
save_plots(drop_results, "Dropout Comparison", "drop")

print("Finished! All 6 graphs saved as PNG files.")