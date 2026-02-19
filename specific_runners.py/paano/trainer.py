import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from vus.metrics import get_metrics
from vus.utils.utility import get_list_anomaly

from model import create_memory_bank, calculate_anomaly_score

class PAanoTrainer():
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.device = device
    
        self.offset = torch.tensor([*range(-config.radius, 0), *range(1, config.radius+1)], dtype=torch.long)
        self.num_iter = config.num_iter
        self.lambda_weight = config.lambda_weight
        self.pretext_steps = config.patch_size
        self.temperature = config.temperature
        self.n_rand_patches = config.n_rand_patches
        self.lr = config.lr
        self.final_lr = config.lr/10
        self.top_k = config.top_k
        self.num_cores = config.num_cores

        self.current_step=0
        self.end = False

        self.criterion_pretext = nn.BCEWithLogitsLoss(reduction='none')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

    def train_epoch(self, pbar):
        self.model.train()
        epoch_triplet_loss = 0
        epoch_context_loss = 0
        
        for batch in self.trainloader:

            if self.current_step>=self.num_iter:
                self.end=True
                break

            self.current_step+=1

            self.update_lr()

            anchors, indices = batch
            anchors = anchors.to(self.device)
            indices = indices.squeeze()

            cand = indices.view(-1, 1) + self.offset.view(1, -1)
            valid = (cand>=0)&(cand<self.n_patches) # potential neighboors.
            
            noise = torch.rand_like(cand.float())
            score = torch.where(valid, noise, torch.full_like(noise, -1))
            choice = torch.argmax(score, dim=-1)
            pos_idx = cand.gather(1, choice.view(-1, 1)).squeeze(1)
            none_valid = valid.sum(dim=1)==0
            if none_valid.any():
                print('okay in fact it happened sometimes')
                pos_idx[none_valid] = indices[none_valid] # if not valid pos is itself (imo never happened)
            positives = torch.stack([self.train_patches[i] for i in pos_idx.tolist()], dim=0)

            if self.current_step < (self.num_iter/10):
                current_lambda_pretext = self.lambda_weight*(1-(self.current_step/(self.num_iter/10)))
            else:
                current_lambda_pretext=0

            if current_lambda_pretext>0:
                pretext_patches = []
                pretext_valid_mask = []

                tgt = indices - self.pretext_steps
                pre_mask = (tgt>=0)&(tgt<self.n_patches)
                tgt_clamped = tgt.clamp(0, self.n_patches-1)

                for i in range(indices.size(0)):
                    if pre_mask[i]:
                        pretext_patches.append(self.train_patches[tgt_clamped[i].item()].unsqueeze(0))
                        pretext_valid_mask.append(True)
                    else:
                        pretext_patches.append(torch.zeros_like(self.train_patches[0].unsqueeze(0)))
                        pretext_valid_mask.append(False)
                        
                pretext_patches = torch.cat(pretext_patches, dim=0)
                pretext_valid_mask = torch.tensor(pretext_valid_mask, dtype=torch.bool, device=self.device)

                all_patches = torch.cat([anchors, positives, pretext_patches], dim=0)
                all_embeddings = self.model.embedding(all_patches)

                h_anchors = all_embeddings[:anchors.size(0)]
                h_pos = all_embeddings[anchors.size(0):2*anchors.size(0)]
                h_pretext = all_embeddings[anchors.size(0)*2:]

            else:
                pretext_patches=None
                pretext_valid_mask=None

                all_patches = torch.cat([anchors, positives], dim=0)
                all_embeddings = self.model.embedding(all_patches)
                h_anchors = all_embeddings[:anchors.size(0)]
                h_pos = all_embeddings[anchors.size(0):]

            z_anchor = self.model.projection(h_anchors)
            z_pos = self.model.projection(h_pos)

            z_anchor = F.normalize(z_anchor, dim=-1)
            z_pos = F.normalize(z_pos, dim=-1)

            sim_ap = (z_anchor @ z_pos.T)/self.temperature
            pos_sims = sim_ap.diag() # we want this to be high (similarity between an anchor and a neighbour)

            sim_ap_f = sim_ap.clone()
            sim_ap_f.diagonal().fill_(-float('inf'))
            neg_dist = 1 - sim_ap_f
            hard_neg_dists, _ = torch.max(neg_dist, dim=1)
            
            pos_dist = 1 - pos_sims
            triplet_loss = F.relu(pos_dist-hard_neg_dists+0.5).mean()

            if current_lambda_pretext>0:
                h_pre = h_pretext[pretext_valid_mask] # keeping only those possible ie not the first patch size patches if stride 1
                h_anchor_pre = h_anchors[pretext_valid_mask]
                h_concat_pre = torch.cat([h_anchor_pre, h_pre], dim=1)

                all_indices = torch.arange(len(anchors), device=self.device)
                anchor_indices = all_indices.repeat_interleave(self.n_rand_patches)
                rand_offsets = torch.randint(1, len(anchors), (len(anchors)*self.n_rand_patches,), device=self.device)
                unadj_indices = (anchor_indices + rand_offsets)%len(anchors) ## retrieve sample in the same batch for fake samples

                h_unadj = h_anchors[unadj_indices]
                h_anchor_unadj = h_anchors.repeat_interleave(self.n_rand_patches, dim=0)
                h_concat_unadj = torch.cat([h_anchor_unadj, h_unadj], dim=1)

                all_pretext_features = torch.cat([h_concat_pre, h_concat_unadj], dim=0)
                all_pretext_labels = torch.cat([
                    torch.ones(h_concat_pre.size(0), device=self.device), 
                    torch.zeros(h_concat_unadj.size(0), device=self.device)
                ], dim=0)

                pretext_outputs = self.model.classification_head(all_pretext_features).squeeze(1)
                pretext_loss_all = self.criterion_pretext(pretext_outputs, all_pretext_labels)

                loss_pre = pretext_loss_all[:h_anchor_pre.size(0)].mean()
                loss_unadj = pretext_loss_all[h_anchor_pre.size(0):].mean()
                pretext_loss = loss_pre+loss_unadj
            else:
                pretext_loss = torch.tensor(0, device=self.device)
            final_loss = triplet_loss + current_lambda_pretext*pretext_loss

            self.optimizer.zero_grad()
            final_loss.backward()
            self.optimizer.step()

            epoch_triplet_loss+=triplet_loss.item()
            epoch_context_loss+=pretext_loss.item()
            pbar.update(1)

        return {"triplet_loss":epoch_triplet_loss, "pretext_loss":epoch_context_loss}

    def fit(self, trainloader, all_patches):

        self.trainloader = trainloader
        self.train_patches = all_patches.to(self.device)
        self.n_patches = len(self.train_patches)
        
        pbar = tqdm(total=self.num_iter, desc="    >> Training")

        while self.current_step<self.num_iter:
            info = self.train_epoch(pbar)
            pbar.set_postfix({
                "pretext_loss": f"{info['pretext_loss']:.4f}", "triplet_loss": f"{info['triplet_loss']:.4f}"
            })
        pbar.close()

        print("Finished training.")
        print("Creating memory bank for anomaly scoring...")
        self.memory_bank, self.memory_indices = create_memory_bank(self.model, self.trainloader, self.device, num_cores=self.num_cores)


    def update_lr(self):
        t = min(self.current_step, self.num_iter)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * t / self.num_iter))
        lr = self.final_lr + (self.lr - self.final_lr) * cosine_factor
        for param_group in self.optimizer.param_groups:
            param_group["lr"]=lr

    def test(self, dataloaders, all_labels):
        slidingWindow = max(int(get_sliding_window(all_labels)), 10)
        results = {}
        all_scores = calculate_anomaly_score(self.model, dataloaders, self.memory_bank, self.device, top_k=self.top_k)
        all_scores = np.asarray(all_scores)
        all_labels = np.asarray(all_labels)
        auc = roc_auc_score(y_true=all_labels, y_score=all_scores)
        results['auc'] = auc
        metrics = get_metrics(all_scores, all_labels, slidingWindow=slidingWindow, metric="vus")
        results['vus_roc'] = metrics['VUS_ROC']
        results['vus_pr'] = metrics['VUS_PR']

        return results

def get_sliding_window(labels):
    return np.median(get_list_anomaly(labels))