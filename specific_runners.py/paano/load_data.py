import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_all_sets(dataset_name, window_size=10):

    if dataset_name=="nyc_taxi" or dataset_name=="ec2_request_latency_system_failure":
        root_dir = "../../data/nab"
        with open(os.path.join(root_dir, "config.json"), "r") as f:
            config = json.load(f)[dataset_name]

        df = pd.read_csv(
            os.path.join(root_dir, f"{dataset_name}.csv"), 
            index_col="timestamp", 
            parse_dates=True
        )
        test_date = pd.to_datetime(config["test_date"])
        test_idx = df.index.searchsorted(test_date, side="left")

        train = df.iloc[:test_idx].copy()
        test = df.iloc[test_idx:].copy()

        scaler = StandardScaler()
        train["value_normed"] = scaler.fit_transform(train[["value"]])
        test["value_normed"] = scaler.transform(test[["value"]])

        full_data = pd.concat([train["value_normed"], test["value_normed"]]).to_numpy()
        full_labels = np.concatenate([
            np.zeros(len(train)), 
            np.array([1 if timestamp in pd.to_datetime(config["anomaly_dates"]) else 0 for timestamp in test.index])
        ])
        test_labels = full_labels[test_idx:]
        train_data = full_data[:test_idx]
        test_data = full_data[test_idx-window_size+1:]
        return [(train_data, test_data, test_labels)]
    
    elif dataset_name=="swat":
        root='../../data/swat/'

        train_data = np.load(os.path.join(root, 'normal.npy'))[-20000:] # Reduction here

        test_labels = np.load(os.path.join(root, 'attack_label.npy'))[window_size-1:]
        test_data = np.load(os.path.join(root, 'attack.npy'))

        return [(train_data, test_data, test_labels)]
    
    elif dataset_name=="smd":
        root = "../../data/smd/processed"
        machines = [m.split(".")[0]+"_" for m in os.listdir("../../data/smd/train")]
        all_sets = []
        for machine in machines:
            scaler = StandardScaler()
            x_normal = np.load(os.path.join(root, machine+"train.npy"))
            x_normal_scaled = scaler.fit_transform(x_normal)

            x_attack = np.load(os.path.join(root, machine+"test.npy"))
            x_attack_scaled = scaler.transform(x_attack)
            test_labels = np.load(os.path.join(root, machine+"test_label.npy"))[window_size-1:]
            train_data = x_normal_scaled
            test_data = x_attack_scaled
            all_sets.append((train_data, test_data, test_labels))

        return all_sets
    
    elif dataset_name=="msl" or dataset_name=="smap":

        smapfiles = ['P-1', 'S-1', 'E-1', 'E-2', 'E-3', 'E-4', 'E-5', 'E-6', 'E-7', 'E-8', 
                    'E-9', 'E-10', 'E-11', 'E-12', 'E-13', 'A-1', 'D-1', 'P-2', 'P-3', 'D-2', 
                    'D-3', 'D-4', 'A-2', 'A-3', 'A-4', 'G-1', 'G-2', 'D-5', 'D-6', 'D-7', 
                    'F-1', 'P-4', 'G-3', 'T-1', 'T-2', 'D-8', 'D-9', 'F-2', 'G-4', 'T-3', 'D-11',
                    'D-12', 'B-1', 'G-6', 'G-7', 'P-7', 'R-1', 'A-5', 'A-6', 'A-7', 'D-13', 
                    'P-2', 'A-8', 'A-9', 'F-3']

        mslfiles = ['M-6', 'M-1', 'M-2', 'S-2', 'P-10', 'T-4', 'T-5', 'F-7', 'M-3', 'M-4', 
                    'M-5', 'P-15', 'C-1', 'C-2', 'T-12', 'T-13', 'F-4', 'F-5', 'D-14', 
                    'T-9', 'P-14', 'T-8', 'P-11', 'D-15', 'D-16', 'M-7', 'F-8']
        
        root="../../data/nasa"
        ano_file = os.path.join(root, "labeled_anomalies.csv")
        values = pd.read_csv(ano_file)
        values = values[values["spacecraft"]==dataset_name.upper()]
        all_sets = []
        
        filenames_possible = values['chan_id'].values.tolist()

        for filename in (smapfiles if dataset_name=="smap" else mslfiles):
            assert filename in filenames_possible, f"filename must be in {filenames_possible}"

            indices = values[values['chan_id']==filename]['anomaly_sequences'].values[0]
            indices = indices.replace("]", "").replace("[", "").split(', ')
            indices = [int(i) for i in indices]

            normal = np.load(os.path.join(root, "train", f"{filename}.npy"))
            scaler = StandardScaler()
            x_normal_scaled = scaler.fit_transform(normal)

            x_attack = np.load(os.path.join(root, "test", f"{filename}.npy"))
            x_attack_scaled = scaler.transform(x_attack)
            test_labels = np.zeros(len(x_attack_scaled))
            for i in range(0, len(indices), 2):
                test_labels[indices[i]:indices[i+1]] = 1

            all_sets.append((x_normal_scaled, x_attack_scaled, test_labels[window_size-1:]))
        return all_sets

    
if __name__ == "__main__":
    dataset_name = "smap"
    window_size = 10
    sets = load_all_sets(dataset_name, window_size)
    for train_data, test_data, test_labels in sets:
        print(f"Train data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")

        for i in range(len(test_labels)):
            if test_labels[i] == 1:
                print(f"Anomaly at index {i} in test set, timestamp, value: {test_data[i:i+10, :3]}, label: {test_labels[i]}")
                break

