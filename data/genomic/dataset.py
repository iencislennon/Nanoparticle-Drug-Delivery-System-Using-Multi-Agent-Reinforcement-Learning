import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class GenomicDataset(Dataset):
    """
        PyTorch Dataset for genomic data
    """

    def __init__(self, features_df, encoded_sequences):
        """
        arguments:
            features_df: DataFrame with patient features (from feature_engineering.py)
            encoded_sequences: numpy array of encoded sequences (from sequence_encoder.py)
        """
        self.features_df = features_df.copy()
        self.encoded_sequences = encoded_sequences

        # Extract components
        self.patient_ids = features_df['patient_id'].values
        self.labels = features_df['drug_response'].values

        # Clinical features columns
        self.clinical_cols = ['age', 'stage', 'mutation_count', 'sex_female']

        # SNP feature columns
        self.snp_cols = [
            col for col in features_df.columns
            if "mutated" in col or "mutation_count" in col
        ]

        # Convert to numeric
        all_cols = self.snp_cols + self.clinical_cols
        for col in all_cols:
            self.features_df[col] = pd.to_numeric(
                self.features_df[col],
                errors="coerce"
            ).fillna(0)

    # NORMALIZATION METHOD (ADDED)
    
    def _normalize_clinical(self, x):
        """
        Min-Max normalize clinical features to [0, 1]
        """
        min_val = x.min()
        max_val = x.max()
        return (x - min_val) / (max_val - min_val + 1e-8)


    def __len__(self):
        return len(self.features_df)

    def __getitem__(self, idx):
        """
        Get one sample.

        returns a dict with:
            - sequence: (1024, 4)
            - snp_features: (n_snps,)
            - clinical_features: (n_clinical,)
            - label
            - patient_id
        """

        # Sequence tensor
        sequence = torch.tensor(
            self.encoded_sequences[idx],
            dtype=torch.float32
        )

        # SNP FEATURES
        snp_values = self.features_df.iloc[idx][self.snp_cols].values
        snp_values = np.array(
            [float(x) if x is not None else 0.0 for x in snp_values],
            dtype=np.float32
        )
        snp_features = torch.tensor(snp_values, dtype=torch.float32)

        # CLINICAL FEATURES (force convert)
        clinical_values = self.features_df.iloc[idx][self.clinical_cols].values
        clinical_values = np.array(
            [float(x) if x is not None else 0.0 for x in clinical_values],
            dtype=np.float32
        )
        clinical_features = torch.tensor(clinical_values, dtype=torch.float32)

        # NORMALIZATION (uses method above)
        clinical_features = self._normalize_clinical(clinical_features)

        # LABEL
        label = torch.tensor(
            [self.labels[idx]],
            dtype=torch.float32
        )

        return {
            "sequence": sequence,
            "snp_features": snp_features,
            "clinical_features": clinical_features,
            "label": label,
            "patient_id": self.patient_ids[idx]
        }

# Collate function (unchanged)

def genomic_collate_fn(batch):

    sequences = torch.stack([item['sequence'] for item in batch])

    # SNP dynamic padding
    max_snps = max(item['snp_features'].shape[0] for item in batch)
    snp_padded = torch.zeros(len(batch), max_snps)
    snp_mask = torch.zeros(len(batch), max_snps, dtype=torch.bool)

    for i, item in enumerate(batch):
        length = item['snp_features'].shape[0]
        snp_padded[i, :length] = item['snp_features']
        snp_mask[i, :length] = True

    clinical = torch.stack([item['clinical_features'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    patient_ids = [item['patient_id'] for item in batch]

    return {
        "sequence": sequences,
        "snp_features": snp_padded,
        "snp_mask": snp_mask,
        "clinical_features": clinical,
        "label": labels,
        "patient_ids": patient_ids
    }


# # TESTING SCRIPT
# if __name__ == "__main__":
#     print("=" * 60)
#     print("GENOMIC DATASET TEST")
#     print("=" * 60)

#     # Load data
#     print("\n1. Loading data...")
#     features_df = pd.read_csv('data/genomic/processed/patient_features.csv')
#     encoded_sequences = np.load('data/genomic/processed/sequences_onehot.npy')

#     print(f"Features: {features_df.shape}")
#     print(f"Sequences: {encoded_sequences.shape}")

#     # Create dataset
#     print("\n2. Creating dataset...")
#     dataset = GenomicDataset(features_df, encoded_sequences)
#     print(f"Dataset size: {len(dataset)}")

#     # Test sample
#     print("\n3. Testing __getitem__...")
#     sample = dataset[0]
#     print(f"   Sequence shape: {sample['sequence'].shape}")
#     print(f"   SNP features shape: {sample['snp_features'].shape}")
#     print(f"   Clinical features shape: {sample['clinical_features'].shape}")
#     print(f"   Label: {sample['label'].item():.3f}")
#     print(f"   Patient ID: {sample['patient_id']}")

#     # DataLoader test
#     print("\n4. Creating DataLoader...")
#     dataloader = DataLoader(
#         dataset,
#         batch_size=8,
#         shuffle=True,
#         collate_fn=genomic_collate_fn,
#         num_workers=0
#     )
#     print(f"DataLoader created with {len(dataloader)} batches")

#     print("\n5. Testing DataLoader...")
#     batch = next(iter(dataloader))

#     print(f"   Sequence batch: {batch['sequence'].shape}")
#     print(f"   SNP features batch: {batch['snp_features'].shape}")
#     print(f"   SNP mask batch: {batch['snp_mask'].shape}")
#     print(f"   Clinical batch: {batch['clinical_features'].shape}")
#     print(f"   Labels batch: {batch['label'].shape}")
#     print(f"   Patient IDs: {len(batch['patient_ids'])}")

#     print("\n" + "=" * 60)
#     print("DATASET TEST COMPLETE")
#     print("=" * 60)
