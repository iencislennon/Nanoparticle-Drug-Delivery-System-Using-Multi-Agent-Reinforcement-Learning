"""Load the real TCGA-BRCA PanCancer Atlas 2018 study from cBioPortal.

Writes into data/genomic/raw/:
  clinical_data.csv          one row per patient (age, stage, subtype, survival, mutation count, sex, race)
  tcga_brca_mutations.csv    real somatic mutations from data_mutations.txt (MAF columns)
  key_snps.csv               small curated list of pharmacogenomic SNPs

Usage:
  python data/genomic/dataset_loader.py                      # download the study archive
  python data/genomic/dataset_loader.py --archive path.tar.gz  # use a local copy

No random data is generated. DNA sequence windows are NOT produced here: they need a
reference genome (GRCh37 for this study) and are a separate step.
"""
import argparse
import io
import os
import tarfile
import urllib.request

import pandas as pd

STUDY = "brca_tcga_pan_can_atlas_2018"
URL = f"https://datahub.assets.cbioportal.org/{STUDY}.tar.gz"
RAW_DIR = "data/genomic/raw"
MAF_COLUMNS = [
    "Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification",
    "Chromosome", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2",
]


def open_archive(path):
    if path:
        return tarfile.open(path, "r:gz")
    print(f"downloading {URL}")
    with urllib.request.urlopen(URL) as resp:
        return tarfile.open(fileobj=io.BytesIO(resp.read()), mode="r:gz")


def read_table(tar, name):
    member = next(m for m in tar.getmembers() if m.name.endswith("/" + name) or m.name == name)
    return pd.read_csv(tar.extractfile(member), sep="\t", comment="#", low_memory=False)


def build_clinical(tar):
    patient = read_table(tar, "data_clinical_patient.txt")
    sample = read_table(tar, "data_clinical_sample.txt")
    df = patient.merge(sample, on="PATIENT_ID", how="inner")
    return pd.DataFrame({
        "Patient_ID": df["PATIENT_ID"],
        "Sample_ID": df["SAMPLE_ID"],
        "Age": df["AGE"],
        "Stage": df["AJCC_PATHOLOGIC_TUMOR_STAGE"],
        "Subtype": df["SUBTYPE"],
        "Survival_Months": df["OS_MONTHS"],
        "Survival_Status": df["OS_STATUS"],
        "Mutation_Count": None,
        "Sex": df["SEX"],
        "Race": df["RACE"],
    })


def build_mutations(tar):
    maf = read_table(tar, "data_mutations.txt")
    out = maf[MAF_COLUMNS].rename(columns={"Tumor_Seq_Allele2": "Tumor_Seq_Allele"})
    out.insert(2, "Patient_ID", out["Tumor_Sample_Barcode"].str[:12])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", help="local path to the cBioPortal study .tar.gz")
    args = ap.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    with open_archive(args.archive) as tar:
        clinical = build_clinical(tar)
        mutations = build_mutations(tar)

    counts = mutations.groupby("Patient_ID").size()
    clinical["Mutation_Count"] = clinical["Patient_ID"].map(counts).fillna(0).astype(int)
    clinical.to_csv(f"{RAW_DIR}/clinical_data.csv", index=False)
    mutations.to_csv(f"{RAW_DIR}/tcga_brca_mutations.csv", index=False)
    pd.DataFrame({
        "Gene": ["CYP2D6", "CYP2D6", "BRCA1", "BRCA2", "TP53", "ABCB1"],
        "rsID": ["rs1065852", "rs16947", "rs1799966", "rs144848", "rs1042522", "rs1045642"],
        "Clinical_Significance": ["Drug_metabolism", "Drug_metabolism", "Pathogenic", "Pathogenic",
                                  "Cancer_risk", "Drug_transport"],
    }).to_csv(f"{RAW_DIR}/key_snps.csv", index=False)

    print(f"patients: {clinical['Patient_ID'].nunique()}")
    print(f"mutations: {len(mutations)} from {mutations['Patient_ID'].nunique()} patients")
    print(f"top mutated genes: {mutations['Hugo_Symbol'].value_counts().head(5).to_dict()}")
    print(clinical["Subtype"].value_counts())


if __name__ == "__main__":
    main()
