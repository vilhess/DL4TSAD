import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader  

class MGABData(Dataset):
    def __init__(
        self, 
        root_dir="data/mgab", 
        dataset_number="1", 
        window_size=10,
        mode="train",
    ):
        dataset_number = str(dataset_number)
        assert dataset_number in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], "Dataset number must be between 1 and 10"
        self.root_dir = root_dir
        self.dataset_number = dataset_number
        self.window_size = window_size
        self.mode = mode

        self.df = pd.read_csv(
            os.path.join(self.root_dir, f"{self.dataset_number}.csv"), 
        )
        self.df.drop(columns=['Unnamed: 0'], inplace=True)
        anomalies = self.df["is_anomaly"].values

        test_idx = 25_000 # fixed test index for MGAB datasets (no anomaly in train set)

        train = self.df.iloc[:test_idx].copy()
        test = self.df.iloc[test_idx:].copy()

        scaler = StandardScaler()
        train["value_normed"] = scaler.fit_transform(train[["value"]])
        test["value_normed"] = scaler.transform(test[["value"]])

        self.df = pd.DataFrame(
            pd.concat([train, test], axis=0, ignore_index=True)
        ).reset_index(drop=True)
        self.df['is_anomaly'] = anomalies

        if self.mode == "train":
            self.data = self.df.iloc[:test_idx]
            self.data = self.data.reset_index(drop=True)
        elif self.mode == "test":
            self.data = self.df.iloc[test_idx - self.window_size:]
            self.data = self.data.reset_index(drop=True)

    def __len__(self):
        return len(self.data) - self.window_size
    
    def __getitem__(self, index):
        start = index
        end = index + self.window_size

        x = self.data['value_normed'].values[start:end+1]
        target = self.data['is_anomaly'].values[end]

        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        target = target
        return x_tensor, target
    
def get_loaders(root_dir="data/mgab", dataset_number="1", window_size=10, batch_size=32):
    trainset = MGABData(root_dir=root_dir, dataset_number=dataset_number, window_size=window_size, mode="train")
    testset = MGABData(root_dir=root_dir, dataset_number=dataset_number, window_size=window_size, mode="test")

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=21)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=21)

    return trainloader, testloader