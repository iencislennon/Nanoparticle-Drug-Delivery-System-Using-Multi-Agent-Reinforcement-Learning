import numpy as np
import pandas as pd
from collections import Counter


class SequenceEncoder:
    """
    Encode DNA sequences for machine learning
    """
    
    def __init__(self, seq_length=1024):
        """
        arguments:
            seq_length: Fixed sequence length
        """
        self.seq_length = seq_length
        
        # Nucleotide mappings
        self.nuc_to_int = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}
        self.int_to_nuc = {v: k for k, v in self.nuc_to_int.items()}
        
        # K-mer vocabulary (for 3-mers)
        self.kmer_vocab = self._generate_kmer_vocab(k=3)
        self.kmer_to_int = {kmer: i for i, kmer in enumerate(self.kmer_vocab)}
    
    def _generate_kmer_vocab(self, k=3):
        """Generate all possible k-mers"""
        from itertools import product
        bases = ['A', 'T', 'C', 'G']
        kmers = [''.join(p) for p in product(bases, repeat=k)]
        return kmers
    
    def one_hot_encode(self, sequence):
        """
        One-hot encode DNA sequence
        
        arguments:
            sequence: DNA string
        
        returns:
            numpy array of shape (seq_length, 4)
        
        Example:
            'ATCG' →
            [[1, 0, 0, 0],  # A
             [0, 1, 0, 0],  # T
             [0, 0, 1, 0],  # C
             [0, 0, 0, 1]]  # G
        """
        sequence = sequence.upper()[:self.seq_length]
        
        # Pad if necessary
        if len(sequence) < self.seq_length:
            sequence = sequence + 'N' * (self.seq_length - len(sequence))
        
        # Create one-hot matrix
        one_hot = np.zeros((self.seq_length, 4))
        
        for i, nuc in enumerate(sequence):
            if nuc in self.nuc_to_int:
                idx = self.nuc_to_int[nuc]
                if idx < 4:  # Ignore N
                    one_hot[i, idx] = 1
        
        return one_hot
    
    def integer_encode(self, sequence):
        """
        Convert sequence to integers
        
        arguments:
            sequence: DNA string
        
        returns:
            numpy array of shape (seq_length,)
        
        Example:
            'ATCG' → [0, 1, 2, 3]
        """
        sequence = sequence.upper()[:self.seq_length]
        
        # Pad if necessary
        if len(sequence) < self.seq_length:
            sequence = sequence + 'N' * (self.seq_length - len(sequence))
        
        # Convert to integers
        encoded = np.array([self.nuc_to_int.get(nuc, 4) for nuc in sequence])
        
        return encoded
    
    def kmer_encode(self, sequence, k=3):
        """
        Encode sequence as k-mer frequencies
        
        arguments:
            sequence: DNA string
            k: k-mer size
        
        returns:
            numpy array of shape (4^k,) with k-mer counts
        
        """
        sequence = sequence.upper()
        
        # Extract k-mers
        kmers = []
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            if 'N' not in kmer:  # Skip k-mers with N
                kmers.append(kmer)
        
        # Count frequencies
        kmer_counts = Counter(kmers)
        
        # Create frequency vector
        freq_vector = np.zeros(len(self.kmer_vocab))
        total_kmers = len(kmers)
        
        if total_kmers > 0:
            for kmer, count in kmer_counts.items():
                if kmer in self.kmer_to_int:
                    idx = self.kmer_to_int[kmer]
                    freq_vector[idx] = count / total_kmers
        
        return freq_vector
    
    def kmer_tokenize(self, sequence, k=3, stride=1):
        """
        Convert sequence to k-mer tokens (like words)
        
        argments:
            sequence: DNA string
            k: k-mer size
            stride: step size (1 = overlapping, k = non-overlapping)
        
        returns:
            list of k-mer token indices
        
        example:
            'ATCGATCG' with k=3, stride=1 →
            ['ATC', 'TCG', 'CGA', 'GAT', 'ATC', 'TCG']
            → [token_id_ATC, token_id_TCG, ...]
        """
        sequence = sequence.upper()
        
        tokens = []
        for i in range(0, len(sequence) - k + 1, stride):
            kmer = sequence[i:i+k]
            if 'N' not in kmer and kmer in self.kmer_to_int:
                tokens.append(self.kmer_to_int[kmer])
        
        return tokens
    
    def batch_encode(self, sequences, method='one_hot'):
        """
        Encode multiple sequences
        
        arguments:
            sequences: list of DNA strings
            method: 'one_hot', 'integer', or 'kmer'
        
        returns:
            numpy array of encoded sequences
        """
        if method == 'one_hot':
            encoded = np.array([self.one_hot_encode(seq) for seq in sequences])
        elif method == 'integer':
            encoded = np.array([self.integer_encode(seq) for seq in sequences])
        elif method == 'kmer':
            encoded = np.array([self.kmer_encode(seq) for seq in sequences])
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return encoded


# # ===== MAIN SCRIPT =====
# if __name__ == "__main__":
#     print("=" * 60)
#     print("DNA SEQUENCE ENCODING")
#     print("=" * 60)
    
#     # Load sequences
#     print("\n1. Loading DNA sequences...")
#     sequences_df = pd.read_csv('data/genomic/raw/dna_sequences.csv')
#     print(f"   ✓ Loaded {len(sequences_df)} sequences")
    
#     # Create encoder
#     encoder = SequenceEncoder(seq_length=1024)
    
#     # Test encoding methods
#     print("\n2. Testing encoding methods...")
    
#     test_seq = sequences_df.iloc[0]['Sequence']
#     print(f"\nTest sequence (first 50 bases):")
#     print(f"  {test_seq[:50]}...")
    
#     # One-hot encoding
#     print("\n--- One-Hot Encoding ---")
#     one_hot = encoder.one_hot_encode(test_seq)
#     print(f"  Shape: {one_hot.shape}")
#     print(f"  First 5 positions:")
#     for i in range(5):
#         nuc = test_seq[i]
#         encoding = one_hot[i]
#         print(f"    {nuc} → {encoding}")
    
#     # Integer encoding
#     print("\n--- Integer Encoding ---")
#     int_encoded = encoder.integer_encode(test_seq)
#     print(f"  Shape: {int_encoded.shape}")
#     print(f"  First 20 values: {int_encoded[:20]}")
#     print(f"  Decoded: {''.join([encoder.int_to_nuc.get(i, 'N') for i in int_encoded[:20]])}")
    
#     # K-mer encoding
#     print("\n--- K-mer Encoding (3-mers) ---")
#     kmer_encoded = encoder.kmer_encode(test_seq, k=3)
#     print(f"  Shape: {kmer_encoded.shape}")
#     print(f"  Non-zero k-mers: {np.sum(kmer_encoded > 0)}")
#     print(f"  Top 5 k-mers:")
#     top_indices = np.argsort(kmer_encoded)[-5:][::-1]
#     for idx in top_indices:
#         kmer = encoder.kmer_vocab[idx]
#         freq = kmer_encoded[idx]
#         print(f"    {kmer}: {freq:.4f}")
    
#     # K-mer tokenization
#     print("\n--- K-mer Tokenization ---")
#     tokens = encoder.kmer_tokenize(test_seq[:20], k=3, stride=1)
#     print(f"  Sequence: {test_seq[:20]}")
#     print(f"  Tokens: {tokens}")
#     print(f"  Decoded k-mers: {[encoder.kmer_vocab[t] for t in tokens]}")
    
#     # Batch encoding
#     print("\n3. Batch encoding all sequences...")
#     sequences = sequences_df['Sequence'].values
    
#     print("\n   Encoding with one-hot...")
#     one_hot_batch = encoder.batch_encode(sequences, method='one_hot')
#     print(f"Shape: {one_hot_batch.shape}")
    
#     print("\n   Encoding with integer...")
#     int_batch = encoder.batch_encode(sequences, method='integer')
#     print(f"Shape: {int_batch.shape}")
    
#     print("\n   Encoding with k-mer...")
#     kmer_batch = encoder.batch_encode(sequences, method='kmer')
#     print(f" Shape: {kmer_batch.shape}")
    
#     # Save encodings
#     print("\n4. Saving encoded sequences...")
    
#     # Save one-hot as npy (binary format, more efficient)
#     np.save('data/genomic/processed/sequences_onehot.npy', one_hot_batch)
#     print("Saved: sequences_onehot.npy")
    
#     np.save('data/genomic/processed/sequences_integer.npy', int_batch)
#     print("Saved: sequences_integer.npy")
    
#     np.save('data/genomic/processed/sequences_kmer.npy', kmer_batch)
#     print("Saved: sequences_kmer.npy")
    
#     # Save encoder config
#     import json
#     config = {
#         'seq_length': encoder.seq_length,
#         'nuc_to_int': encoder.nuc_to_int,
#         'vocab_size_kmer': len(encoder.kmer_vocab),
#         'kmer_size': 3
#     }
    
#     with open('data/genomic/processed/encoder_config.json', 'w') as f:
#         json.dump(config, f, indent=2)
#     print("Saved: encoder_config.json")
    
#     # Statistics
#     print("\n" + "=" * 60)
#     print("ENCODING STATISTICS:")
#     print("=" * 60)
    
#     print(f"\nSequences encoded: {len(sequences)}")
#     print(f"Sequence length: {encoder.seq_length} bp")
    
#     print(f"\nOne-hot encoding:")
#     print(f"  Shape: {one_hot_batch.shape}")
#     print(f"  Memory: {one_hot_batch.nbytes / 1024 / 1024:.2f} MB")
    
#     print(f"\nInteger encoding:")
#     print(f"  Shape: {int_batch.shape}")
#     print(f"  Memory: {int_batch.nbytes / 1024:.2f} KB")
    
#     print(f"\nK-mer encoding:")
#     print(f"  Shape: {kmer_batch.shape}")
#     print(f"  Vocabulary size: {len(encoder.kmer_vocab)} (3-mers)")
#     print(f"  Memory: {kmer_batch.nbytes / 1024:.2f} KB")
    
#     print("\n" + "=" * 60)
#     print("SEQUENCE ENCODING COMPLETE!")
#     print("=" * 60)