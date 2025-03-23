import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scipy.stats import beta
from tqdm import trange
from typing import Tuple, List, Optional

def subsample_instances(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices

class MergedDataset(Dataset):

    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset):

        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None

    def __getitem__(self, item):

        if item < len(self.labelled_dataset):
            img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1

        else:

            img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0


        return img, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)

def extract_features(model: torch.nn.Module, dataset: Dataset, batch_size: int = 32) -> torch.Tensor:
    """Extract features from a dataset using a pretrained model."""
    model.eval()
    features = []
    uq_idxs = []
    
    # import ipdb; ipdb.set_trace()
    with torch.no_grad():
        for i in trange(0, len(dataset), batch_size):
            batch_indices = range(i, min(i + batch_size, len(dataset)))
            batch_images = []
            batch_uq_idxs = []
            for ind in batch_indices:
                batch_images.append(dataset[ind][0][0])
                uq_idxs.append(dataset[ind][2])
            batch_images = torch.stack(batch_images).cuda()
            batch_features = model(batch_images)
            features.append(batch_features.cpu().detach())
    
    return torch.cat(features, dim=0), uq_idxs

def cosine_similarity_matrix(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between two sets of feature vectors."""
    return F.cosine_similarity(A.unsqueeze(1), B.unsqueeze(0), dim=-1)

class BinningDataSelector:
    """Implements binning-based data selection for category discovery."""
    
    def __init__(self, 
                 num_clusters: int = 10,
                 threshold_low: float = 0.2,
                 threshold_high: float = 0.8,
                 random_state: int = 42):
        self.num_clusters = num_clusters
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    
    def select_data(self, 
                   labeled_features: torch.Tensor,
                   unlabeled_features: torch.Tensor) -> torch.Tensor:
        """Select labeled data based on similarity to unlabeled clusters."""
        # Cluster unlabeled data
        cluster_labels = self.kmeans.fit_predict(unlabeled_features.numpy())
        
        # Compute cluster means
        cluster_means = torch.stack([
            unlabeled_features[cluster_labels == i].mean(dim=0) 
            for i in range(self.num_clusters)
        ])
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity_matrix(labeled_features, cluster_means)
        
        # Select samples based on similarity thresholds
        max_similarities = similarity_matrix.max(dim=1).values
        selected_indices = (max_similarities > self.threshold_low) & \
                          (max_similarities < self.threshold_high)
        
        return selected_indices

class BetaWeightingDataSelector:
    """Implements beta-distribution based soft weighting for category discovery."""
    
    def __init__(self, alpha: float = 5.0, beta_param: float = 5.0):
        self.alpha = alpha
        self.beta_param = beta_param
    
    def compute_weights(self, 
                       labeled_features: torch.Tensor,
                       unlabeled_features: torch.Tensor) -> torch.Tensor:
        """Compute weights for labeled data using beta distribution."""
        # Compute similarity to farthest unlabeled point
        similarities = cosine_similarity_matrix(labeled_features, unlabeled_features)
        similarity_scores = similarities.min(dim=1).values
        
        # Convert to numpy for scipy compatibility
        scores_np = similarity_scores.numpy()
        
        # Compute beta distribution weights
        rv = beta(self.alpha, self.beta_param)
        weights = rv.pdf(scores_np,)
        weights = torch.tensor(weights / weights.max())  # Normalize to [0,1]
        
        return weights

def weighted_loss(loss: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Apply weights to a loss tensor."""
    return (loss * weights).mean()
