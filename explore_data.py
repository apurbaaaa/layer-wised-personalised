"""Quick data exploration script."""
import pandas as pd

df = pd.read_csv("data/ISIC_2019_Training_Metadata.csv")
print("=== Training Metadata ===")
print(df.dtypes)
print()
print("age_approx unique:", sorted(df["age_approx"].dropna().unique()))
print("age_approx NaN:", df["age_approx"].isna().sum())
print()
print("anatom_site_general unique:", sorted(df["anatom_site_general"].dropna().unique()))
print("anatom_site_general NaN:", df["anatom_site_general"].isna().sum())
print()
print("sex unique:", sorted(df["sex"].dropna().unique()))
print("sex NaN:", df["sex"].isna().sum())
print()
print("lesion_id NaN:", df["lesion_id"].isna().sum())
print()

gt = pd.read_csv("data/ISIC_2019_Training_GroundTruth.csv")
classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
print("Class distribution:")
for c in classes:
    print(f"  {c}: {int(gt[c].sum())}")
print(f"  Total: {len(gt)}")
