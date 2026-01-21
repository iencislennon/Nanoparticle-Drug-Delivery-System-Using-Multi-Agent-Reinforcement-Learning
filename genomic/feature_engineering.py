import pandas as pd
import numpy as np
from collections import Counter
import re
#get
class Genomic_Feature_extractor:
    """
    Extract features for ML models
    """

    def __init__(self):
        self.drug_metabolism_genes = ['CYP2D6', 'CYP2C9', 'CYP2C19', 'CYP3A4']
        self.dna_repair_genes = ['BRCA1', 'BRCA2', 'TP53', 'ATM', 'CHEK2']
        self.drug_transport_genes  = ['ABCB1', 'SLCO1B1', 'ABCG2']
        
        # k-mer size
        self.k_mer_size = 3
    def extract_snp_features(self, patient_mutations, key_snps):
        """
        Docstring для extract_snp_features
            arguments:
                patient_mutations: DataFrame of mutations for patient
                key_snps: DataFrame of important SNPs
        
            returns:
                dict of SNP features

        """
        snp_features = {}

        for gene in self.drug_metabolism_genes+self.dna_repair_genes+self.drug_transport_genes:
            gene_mutatinos = patient_mutations[patient_mutations['Hugo_Symbol'] == gene]

            if len(gene_mutatinos) > 0:
                snp_features[f"{gene}_mutated"] = 1
                snp_features[f"{gene}_mutation_count"] = len(gene_mutatinos)
            else:
                snp_features[f'{gene}_mutated'] = 0
                snp_features[f"{gene}_mutation_count"] = 0
        return snp_features
    
    def extract_sequence_features(self, sequence):
        """
            Extract sequence-based features from DNA
        
            arguments:
                sequence: DNA sequence string
        
            returns:
                dict of sequence features
        """
        sequence = sequence.upper()
        seq_len = len(sequence)

        features = {}

        # GC 
        gc_count = sequence.count('G') + sequence.count('C')
        features['gc_content'] = gc_count / seq_len if seq_len > 0 else 0

        # individual nucleotide frequencies
        features['a_freq'] = sequence.count('A') / seq_len if seq_len >0 else 0 
        features['t_freq'] = sequence.count('T') / seq_len if seq_len >0 else 0
        features['c_freq'] = sequence.count('C') / seq_len if seq_len >0 else 0
        features['g_freq'] = sequence.count('G') / seq_len if seq_len >0 else 0

        # K-mer frequiences (3-mers)
        kmers = self._get_kmers(sequence, k = 3)
        kmer_counts = Counter(kmers)

        # top k-mers as features
        top_kmers= ['ATG', 'TGC', 'GCA', 'CAT', 'TAC', 'ACG'] # Common important k-mers
        for kmer in top_kmers:
            features[f"kmer_{kmer}"] = kmer_counts.get(kmer, 0) / len(kmers) if len(kmers) > 0 else 0

        # repeat detection (simple homoplayers)
        features['a_repeats'] = self._count_repeats(sequence, 'A')
        features['t_repeats'] = self._count_repeats(sequence, 'T')
        features['c_repeats'] = self._count_repeats(sequence, 'C')
        features['g_repeats'] = self._count_repeats(sequence, 'G')

        return features
    def _get_kmers(self, sequence, k=3):
        """
            Extract all k-mers from sequence
            arguments:
                sequence: DNA string
                k: k-mer size
            returns:
                list of k_mers
        """
        kmers = []
        for i in range(len(sequence) - k+1):
            kmer = sequence[i:i+k]
            # only include valid k-mers; no Ns
            if 'N' not in kmer:
                kmers.append(kmer)
        return kmers
    def _count_repeats(self, sequence, base, min_length=3):
        """ 
            count homopolymer repeats( e.g AAA, TTTT)
            arguments:
                sequence: DNA string
                base: (A,T,C,G)
                min_length: minimum repeat length
            returns:
                count of repeats
        """
        pattern = f"{base}{{{min_length},}}"
        matches = re.findall(pattern, sequence)
        
        return len(matches)
    def extract_clinical_features(self, clinical_row):
        """
        arguments:
            clinical row - pandas Series with clinical data
        returns:
            dict of clinical features
        """
        features = {}

        features['age'] = clinical_row['Age']

        #converting to numeric
        stage_mapping = {
            'STAGE I': 1,
            'STAGE IA': 1,
            'STAGE IB': 1,
            'STAGE II': 2,
            'STAGE IIA': 2,
            'STAGE IIB': 2,
            'STAGE III': 3,
            'STAGE IIIA': 3,
            'STAGE IIIB': 3,
            'STAGE IIIC': 3,
            'STAGE IV': 4,
            'STAGE X': 0  # Unknown
        }        
        stage_str = str(clinical_row['Stage'])
        features['stage'] = stage_mapping.get(stage_str,0)

        # one hot encoding
        subtypes = ['BRCA_LumA', 'BRCA_LumB', 'BRCA_Her2', 'BRCA_Basal', 'BRCA_Normal']
        subtype = str(clinical_row['Subtype'])
        for st in subtypes:
            features[f"subtype_{st}"] = 1 if subtype == st else 0
        
        features['mutation_count'] = clinical_row['Mutation_Count']
        features['sex_female'] = 1 if clinical_row['Sex'] == 'Female' else 0
        features['sex_male'] = 1 if clinical_row['Sex'] == 'Male' else 0

        return features
    def create_target_variable(self, clinical_row):
        """
        Create drug response target variable from survival data
        
        Good responders = long survival + living
        Poor responders = short survival or deceased
        
        arguments:
            clinical_row: pandas Series with clinical data
        
        returns:
            float between 0-1 (drug response score)
        """

        survival_months = clinical_row['Survival_Months']
        survival_status = clinical_row['Survival_Status']

        time_score = min(survival_months/120.0, 1.0) #Base score on survival time (normalize to 0-1)

        #adjust based on status
        if '0:LIVING' in survival_status or 'LIVING' in survival_status:
            # if alive better response 
            response_score = 0.5 + (time_score*0.5) # 0.5 to 1.0
        else:
            # deceased = poorer response
            response_score = time_score * 0.5 # 0 to 0.5
        return response_score
    def extract_all(self, patient_id, clinical_data, mutations_data, sequences_data):
        """
            extract complete feature set for a patient
            arguments:
                patient_id: Patient ID
                clinical_data: DataFrame of clinical data
                mutations_data: DataFrame of mutations
                sequences_data: DataFrame of DNA sequences
            returns:
                dict with all features
        """
        
        clinical_row = clinical_data[clinical_data['Patient_ID'] == patient_id].iloc[0]
        patient_mutations = mutations_data[mutations_data['Tumor_Sample_Barcode'] == patient_id]
        
        seq_row = sequences_data[sequences_data['Patient_ID'] == patient_id]
        if len(seq_row) > 0:
            sequence = seq_row.iloc[0]['Sequence']
        else:
            sequence = ''.join(np.random.choice(['A','T','C','G'], size=1024))
        # extract all feature types
        features = {}
        features['patient_id'] = patient_id
        features['dna_sequence'] = sequence

        # snp features
        snp_features = self.extract_snp_features(patient_mutations, None)
        features.update(snp_features)

        # sequence features
        seq_features = self.extract_sequence_features(sequence)
        features.update(seq_features)

        #clinical features
        clinical_features = self.extract_clinical_features(clinical_row)
        features.update(clinical_features)

        #target variable
        features['drug_response'] = self.create_target_variable(clinical_row)

        return features
# if __name__ == "__main__":
#     print("=" * 60)
#     print("GENOMIC FEATURE ENGINEERING")
#     print("=" * 60)
    
#     # Load data
#     print("\n1. Loading data...")
#     clinical = pd.read_csv('data/genomic/raw/clinical_data.csv')
#     mutations = pd.read_csv('data/genomic/raw/tcga_brca_mutations.csv')
#     sequences = pd.read_csv('data/genomic/raw/dna_sequences.csv')
    
#     print(f"   ✓ Loaded: {len(clinical)} patients")
#     print(f"   ✓ Loaded: {len(mutations)} mutations")
#     print(f"   ✓ Loaded: {len(sequences)} sequences")
    
#     # Create feature extractor
#     extractor = Genomic_Feature_extractor()
    
#     # Extract features for all patients
#     print("\n2. Extracting features...")
#     all_features = []
    
#     # Process first 100 patients (or all with sequences)
#     patient_ids = sequences['Patient_ID'].unique()[:100]
    
#     for i, patient_id in enumerate(patient_ids):
#         if (i + 1) % 20 == 0:
#             print(f"   Processing patient {i+1}/{len(patient_ids)}...")
        
#         try:
#             features = extractor.extract_all(
#                 patient_id,
#                 clinical,
#                 mutations,
#                 sequences
#             )
#             all_features.append(features)
#         except Exception as e:
#             print(f"   ✗ Error for {patient_id}: {e}")
#             continue
    
#     # Convert to DataFrame
#     features_df = pd.DataFrame(all_features)
    
#     print(f"\n   ✓ Extracted features for {len(features_df)} patients")
#     print(f"   ✓ Total features: {len(features_df.columns)}")
    
#     # Save
#     features_df.to_csv('data/genomic/processed/patient_features.csv', index=False)
#     print(f"\n   ✓ Saved: data/genomic/processed/patient_features.csv")
    
#     # Display sample
#     print("\n" + "=" * 60)
#     print("SAMPLE FEATURES (first patient):")
#     print("=" * 60)
    
#     sample = features_df.iloc[0]
#     print(f"\nPatient ID: {sample['patient_id']}")
#     print(f"Age: {sample['age']}")
#     print(f"Stage: {sample['stage']}")
#     print(f"GC Content: {sample['gc_content']:.3f}")
#     print(f"Mutation Count: {sample['mutation_count']}")
#     print(f"Drug Response (target): {sample['drug_response']:.3f}")
    
#     print(f"\nSNP Features:")
#     snp_cols = [c for c in features_df.columns if 'mutated' in c or 'mutation_count' in c]
#     for col in snp_cols[:5]:
#         print(f"  {col}: {sample[col]}")
    
#     print(f"\nSequence Features:")
#     seq_cols = ['gc_content', 'a_freq', 't_freq', 'c_freq', 'g_freq']
#     for col in seq_cols:
#         print(f"  {col}: {sample[col]:.3f}")
    
#     # Statistics
#     print("\n" + "=" * 60)
#     print("FEATURE STATISTICS:")
#     print("=" * 60)
    
#     print(f"\nTarget variable (drug_response):")
#     print(f"  Mean: {features_df['drug_response'].mean():.3f}")
#     print(f"  Std: {features_df['drug_response'].std():.3f}")
#     print(f"  Min: {features_df['drug_response'].min():.3f}")
#     print(f"  Max: {features_df['drug_response'].max():.3f}")
    
#     print(f"\nMutation burden:")
#     print(f"  Mean mutations: {features_df['mutation_count'].mean():.1f}")
#     print(f"  Std: {features_df['mutation_count'].std():.1f}")
    
#     print(f"\nGC content:")
#     print(f"  Mean: {features_df['gc_content'].mean():.3f}")
#     print(f"  Std: {features_df['gc_content'].std():.3f}")
    
#     print("\n" + "=" * 60)
#     print("✅ FEATURE ENGINEERING COMPLETE!")
#     print("=" * 60)


        

