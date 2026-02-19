import torch
import numpy as np

class _tsdataset(torch.utils.data.Dataset):
    def __init__(self, data, indices=None):
        self.data = torch.from_numpy(np.array(data)).float()
        if indices is not None:
            self.indices = torch.from_numpy(np.array(indices)).long().unsqueeze(1)
        else:
            self.indices=None
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        if x.ndim==1:
            x = x.unsqueeze(0).contiguous()
        if self.indices is not None:
            return x, self.indices[index]
        return x, torch.tensor([index])
    
class PatchCreator:
    def __init__(self, L, s, random_seed=None):
        self.L = L
        self.s = s
        if random_seed is not None:
            torch.manual_seed(random_seed)

    def create_patches(self, data):
        num_patches = (len(data)-self.L) // self.s + 1
        patches = [data[i:i+self.L] for i in range(0, len(data)-self.L + 1, self.s)]
        indices = [i for i in range(0, len(data) - self.L + 1, self.s)]
        return patches, indices
    
    def create_dataloaders(self, train_data, test_data, test_labels, batch_size=512):
        train_patches, train_indices = self.create_patches(train_data)
        test_patches, test_indices = self.create_patches(test_data)

        train_patches = [p.T for p in train_patches] if train_patches[0].ndim==2 else train_patches
        test_patches = [p.T for p in test_patches] if test_patches[0].ndim==2 else test_patches

        drop_last = len(train_patches) % batch_size == 1
        trainloader = torch.utils.data.DataLoader(_tsdataset(train_patches, train_indices), batch_size=batch_size, shuffle=True, drop_last=drop_last)
        testloader = torch.utils.data.DataLoader(_tsdataset(test_patches, test_indices), batch_size=batch_size, shuffle=False)
        test_labels = test_labels
        return trainloader, testloader, test_labels

class PaAnoSignalProcessor:
    def __init__(self, train_data, test_data, test_labels, patch_size=64, stride=1):
        self.train_data = train_data
        self.test_data = test_data
        self.test_labels = test_labels
        self.patch_size = patch_size
        self.stride = stride
        self.patch_creator = PatchCreator(patch_size, stride)
    
    def preprocess_to_patches(self, data):
        patches=[]
        for i in range(0, len(data)-self.patch_size+1, self.stride):
            patch = data[i:i+self.patch_size]
            patches.append(patch)
        patches_array = np.array(patches)
        t = torch.tensor(patches_array, dtype=torch.float32)
        if t.ndim==2:
            t = t.unsqueeze(1).contiguous()
        elif t.ndim==3:
            t = t.permute(0, 2, 1).contiguous()
        return t
    
    def get_loaders(self, batch_size=512):
        trainloader, testloader, test_labels = self.patch_creator.create_dataloaders(
            self.train_data, self.test_data, self.test_labels, batch_size=batch_size
        )
        return trainloader, testloader, test_labels
    
    def get_all_patches(self, set="train"):
        
        if set == "train":
            train_patches = self.preprocess_to_patches(self.train_data)
            return train_patches
        elif set == "test":
            test_patches = self.preprocess_to_patches(self.test_data)
            return test_patches
        else:
            all_patches = torch.cat([train_patches, test_patches], dim=0)
            return all_patches