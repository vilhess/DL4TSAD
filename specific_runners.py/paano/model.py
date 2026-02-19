import torch 
import torch.nn as nn 
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans


class RevIN1d(nn.Module):    
    def __init__(self, eps=1e-5, min_sigma=1e-5):
        super().__init__()
        self.eps = eps
        self.min_sigma = min_sigma

        self._mu = None
        self._sigma = None

    @torch.no_grad()
    def _stats(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        sigma = (var+self.eps).sqrt().clamp_min(self.min_sigma)
        return mu, sigma
    
    def norm(self, x):
        self._mu, self._sigma = self._stats(x)
        x_hat = (x-self._mu)/self._sigma
        return x_hat

    def denorm(self, x_hat):
        mu, sigma = self._mu, self._sigma
        return x_hat*sigma + mu

class PatchEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_channels = config['in_dim']
        self.layers = config['layers']
        self.kss = config['kss']
        self.projection_dim = config['projection_dim']      

        self.revin = RevIN1d(eps=1e-5, min_sigma=1e-5)  

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.layers[i-1] if i>0 else in_channels, self.layers[i], kernel_size=self.kss[i], stride=1, padding=self.kss[i]//2, bias=False),
                nn.BatchNorm1d(self.layers[i]), 
                nn.ReLU(inplace=True)
            ) for i in range(len(self.layers))
        ])

        self.fc_embedding = nn.AdaptiveAvgPool1d(output_size=1)
        self.projection_head = nn.Sequential(
            nn.Linear(self.layers[-1], self.projection_dim), 
            nn.ReLU(),
            nn.Linear(self.projection_dim, self.projection_dim)
        )
        self.classification_head = nn.Linear(self.layers[-1]*2, 1)

    def forward(self, x, return_embedding=False, return_projection=False):
        x = self.revin.norm(x)
        for block in self.conv_blocks:
            x = block(x)
        h = self.fc_embedding(x).flatten(start_dim=1)
        if return_embedding:
            return h
        if return_projection:
            return self.projection_head(h)
        raise ValueError("The forward method is not designed to handle classification directly.")
    
    def embedding(self, x):
        return self.forward(x, return_embedding=True, return_projection=False)
    
    def projection(self, h):
        return self.projection_head(h)
    
@torch.no_grad()
def create_memory_bank(model, dataloader, device, num_cores=None):
    model.eval()
    embeddings = []
    indices = []
    for data, indice in dataloader:
        data = data.to(device)
        h = model.embedding(data)
        embeddings.append(h.detach().cpu().float())
        indices.append(indice.detach().cpu())
    embeddings = torch.cat(embeddings, dim=0)
    indices = torch.cat(indices, dim=0)
    num_samples = len(embeddings)

    if num_cores is None:
        return embeddings, indices
    
    if isinstance(num_cores, float):
        k = int(round(num_cores*num_samples))
    else:
        k = int(num_cores)
    
    min_cores_eff = min(500, max(1, num_samples-1))
    num_cores = max(min_cores_eff, min(k, num_samples-1) )
    if num_cores>=num_samples:
        return embeddings, indices
    
    flattened = embeddings.view(num_samples, -1)
    flattened = F.normalize(flattened, p=2, dim=1)

    mbk = MiniBatchKMeans(n_clusters=num_cores, init='k-means++', random_state=42, 
                          batch_size=max(8192, num_cores), max_iter=50, n_init=1, reassignment_ratio=0.01)
    
    mbk.fit(flattened.numpy())

    centers = torch.tensor(mbk.cluster_centers_, dtype=flattened.dtype)
    distances = torch.cdist(flattened, centers, p=2)
    core_indices = torch.argmin(distances, dim=0)
    embeddings = embeddings[core_indices]
    indices = indices[core_indices]
    return embeddings, indices

@torch.inference_mode()
def calculate_anomaly_score(model, test_loader, memory_bank, device, top_k=3):
    model.eval()
    all_scores=[]
    memory_bank=F.normalize(memory_bank.to(device, dtype=torch.float32), dim=1, eps=1e-12)

    for data, _ in test_loader:
        data = data.to(device)
        feats = model.embedding(data)
        feats = torch.nan_to_num(feats, nan=0., posinf=0., neginf=0.)
        feats = F.normalize(feats, dim=1, eps=1e-12)
        feats = torch.nan_to_num(feats, nan=0., posinf=0., neginf=0.)

        sims = feats@memory_bank.T
        sims = torch.nan_to_num(sims, nan=-1, posinf=1, neginf=-1)
        topk_sim, _ = torch.topk(sims, k=top_k, dim=1, largest=True)

        dists = 1-topk_sim
        scores = dists.mean(dim=-1)
        scores = torch.nan_to_num(scores, nan=1., posinf=1., neginf=0.)
        all_scores.extend(scores.cpu().tolist())
    return all_scores