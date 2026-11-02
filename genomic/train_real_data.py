"""Train and evaluate a 5-year-survival classifier on the real TCGA-BRCA cohort.

Run data/genomic/dataset_loader.py first. Label: deceased within 60 months (1) vs alive
with at least 60 months of follow-up (0); patients censored earlier are excluded.
This is a survival proxy, not a measured drug response. Splits are by patient and
stratified; the test set is touched once.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAW = "data/genomic/raw"
GENES = ["TP53", "PIK3CA", "GATA3", "CDH1", "MAP3K1", "BRCA1", "BRCA2", "AKT1", "PTEN", "RB1",
         "ATM", "CHEK2", "ABCB1", "CYP2D6", "CYP2C19", "CYP3A4"]
STAGE = {"I": 1, "II": 2, "III": 3, "IV": 4}
SEED = 0


def stage_num(s):
    s = str(s).upper().replace("STAGE", "").strip().rstrip("ABC")
    return STAGE.get(s, np.nan)


def build():
    clin = pd.read_csv(f"{RAW}/clinical_data.csv").drop_duplicates("Patient_ID")
    mut = pd.read_csv(f"{RAW}/tcga_brca_mutations.csv")
    done = clin["Survival_Status"].astype(str).str.contains("DECEASED")
    clin = clin[(done & (clin["Survival_Months"] < 60)) | (clin["Survival_Months"] >= 60)].copy()
    clin["y"] = (done & (clin["Survival_Months"] < 60)).loc[clin.index].astype(int)

    flags = (mut[mut["Hugo_Symbol"].isin(GENES)]
             .assign(v=1).pivot_table(index="Patient_ID", columns="Hugo_Symbol", values="v", aggfunc="max")
             .reindex(columns=GENES).fillna(0))
    flags.columns = [f"{g}_mut" for g in flags.columns]
    X = clin.set_index("Patient_ID")[["Age"]].copy()
    X["Mutation_Count"] = mut.groupby("Patient_ID").size().reindex(X.index).fillna(0)
    X["stage"] = clin.set_index("Patient_ID")["Stage"].map(stage_num)
    X = X.join(pd.get_dummies(clin.set_index("Patient_ID")["Subtype"]).astype(int))
    X = X.join(flags).fillna({c: 0 for c in flags.columns})
    X = X.fillna(X.median(numeric_only=True))
    y = clin.set_index("Patient_ID")["y"]
    return X, y


def main():
    X, y = build()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    out = {"n_patients": int(len(y)), "positives": int(y.sum()), "n_train": int(len(ytr)), "n_test": int(len(yte)),
           "label": "deceased <60 months vs alive >=60 months follow-up", "models": {}}
    for name, m in models.items():
        cv_auc = roc_auc_score(ytr, cross_val_predict(m, Xtr, ytr, cv=cv, method="predict_proba")[:, 1])
        m.fit(Xtr, ytr)
        out["models"][name] = {"cv_auroc_train": round(float(cv_auc), 3),
                               "test_auroc": round(float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1])), 3)}
    os.makedirs("outputs", exist_ok=True)
    json.dump(out, open("outputs/real_data_metrics.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
