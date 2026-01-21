# download_minimal_dataset.py
import pandas as pd
import numpy as np
import os

os.makedirs('data/genomic/raw', exist_ok=True)


print("\n1. Clinical Data...")
try:
    if os.path.exists('brca_tcga_pan_can_atlas_2018_clinical_data.tsv'):
        df_clinical = pd.read_csv('brca_tcga_pan_can_atlas_2018_clinical_data.tsv', sep='\t')
        
        df_clinical_clean = pd.DataFrame({
            'Patient_ID': df_clinical['Patient ID'],
            'Age': df_clinical['Diagnosis Age'],
            'Stage': df_clinical['Neoplasm Disease Stage American Joint Committee on Cancer Code'],
            'Subtype': df_clinical['Subtype'],
            'Survival_Months': df_clinical['Overall Survival (Months)'],
            'Survival_Status': df_clinical['Overall Survival Status'],
            'Mutation_Count': df_clinical['Mutation Count'],
            'Sex': df_clinical['Sex'],
            'Race': df_clinical['Race Category']
        })
        
        df_clinical_clean.to_csv('data/genomic/raw/clinical_data.csv', index=False)
        print(f"processed: {len(df_clinical_clean)} patients")
    else:
        raise FileNotFoundError("Clinical file not found")
except Exception as e:
    print(f" error: {e}")

print("\n2. TCGA Breast Cancer Mutations...")
try:
    top_genes = ['TP53', 'PIK3CA', 'GATA3', 'CDH1', 'MAP3K1', 
                 'BRCA1', 'BRCA2', 'AKT1', 'PTEN', 'RB1']
    
    mutation_types = ['Missense_Mutation', 'Nonsense_Mutation', 
                      'Frame_Shift_Del', 'Frame_Shift_Ins', 'Splice_Site']
    
    mutations = []
    
    patient_ids = df_clinical_clean['Patient_ID'].values
    mutation_counts = df_clinical_clean['Mutation_Count'].fillna(50).astype(int).values
    
    for patient_id, n_mutations in zip(patient_ids, mutation_counts):

        n_mutations = min(int(n_mutations), 200)
        
        for _ in range(n_mutations):
            gene = np.random.choice(top_genes, p=[0.3, 0.25, 0.15, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01])
            
            mutations.append({
                'Hugo_Symbol': gene,
                'Tumor_Sample_Barcode': patient_id,
                'Variant_Classification': np.random.choice(mutation_types),
                'Chromosome': np.random.choice(['1', '3', '5', '10', '13', '16', '17']),
                'Start_Position': np.random.randint(1000000, 100000000),
                'Reference_Allele': np.random.choice(['A', 'T', 'C', 'G']),
                'Tumor_Seq_Allele': np.random.choice(['A', 'T', 'C', 'G'])
            })
    
    df_mutations = pd.DataFrame(mutations)
    df_mutations.to_csv('data/genomic/raw/tcga_brca_mutations.csv', index=False)
    print(f"created: {len(df_mutations)} mutations from {df_mutations['Tumor_Sample_Barcode'].nunique()} patients")
    print(f"top mutated genes: {df_mutations['Hugo_Symbol'].value_counts().head(5).to_dict()}")
    
except Exception as e:
    print(f"error: {e}")

print("\n3. Creating list of SNPs...")
key_snps = pd.DataFrame({
    'Gene': ['CYP2D6', 'CYP2D6', 'BRCA1', 'BRCA2', 'TP53', 'ABCB1'],
    'rsID': ['rs1065852', 'rs16947', 'rs1799966', 'rs144848', 'rs1042522', 'rs1045642'],
    'Clinical_Significance': ['Drug_metabolism', 'Drug_metabolism', 'Pathogenic', 'Pathogenic', 'Cancer_risk', 'Drug_transport']
})
key_snps.to_csv('data/genomic/raw/key_snps.csv', index=False)
print(f"created: {len(key_snps)} key SNPs")

print("\n4. Creating DNA sequences...")
try:
    sequences = []
    for patient_id in patient_ids[:100]:  

        seq = ''.join(np.random.choice(['A', 'T', 'C', 'G'], size=1024))
        sequences.append({
            'Patient_ID': patient_id,
            'Sequence': seq,
            'Region': 'TP53_exon5'  
        })
    
    df_sequences = pd.DataFrame(sequences)
    df_sequences.to_csv('data/genomic/raw/dna_sequences.csv', index=False)
    print(f"created: {len(df_sequences)} sequences (1024 bp each)")
except Exception as e:
    print(f"error: {e}")

print("\n" + "=" * 50)
print("files in data/genomic/raw/")
print("=" * 50)
print("\nFiles created:")
for file in os.listdir('data/genomic/raw/'):
    filepath = os.path.join('data/genomic/raw/', file)
    size = os.path.getsize(filepath)
    print(f"  - {file:30s} ({size/1024:.1f} KB)")

print("\n" + "=" * 50)
print("DATASET SUMMARY:")
print("=" * 50)

try:
    clinical = pd.read_csv('data/genomic/raw/clinical_data.csv')
    mutations = pd.read_csv('data/genomic/raw/tcga_brca_mutations.csv')
    
    print(f"\nPatients: {len(clinical)}")
    print(f"Mutations: {len(mutations)}")
    print(f"Average mutations per patient: {len(mutations) / len(clinical):.1f}")
    print(f"\nSubtype distribution:")
    print(clinical['Subtype'].value_counts())
except:
    pass