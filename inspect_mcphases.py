import os
import pandas as pd

DATA_DIR = "mcPHASES"   # change if needed
MAX_UNIQUE_TO_PRINT = 30

FILES = [
    "hormones_and_selfreport.csv",
    "sleep.csv",
    "stress_score.csv",
    "heart_rate_variability_details.csv",
    "exercise.csv",
    "active_minutes.csv",
    "computed_temperature.csv",   # or wrist_temperature.csv
    "resting_heart_rate.csv",
]

def inspect_dataframe(name, df):
    print("\n" + "="*80)
    print(f"FILE: {name}")
    print("="*80)

    # Identify non-numeric columns
    non_numeric_cols = [
        col for col in df.columns
        if df[col].dtype == "O"   # object/string
           or str(df[col].dtype).startswith("category")
           or str(df[col].dtype).startswith("bool")
    ]

    if not non_numeric_cols:
        print("No non-numeric columns.\n")
        return

    print(f"Found {len(non_numeric_cols)} non-numeric columns.\n")

    for col in non_numeric_cols:
        print("-" * 60)
        print(f"Column: {col}")
        print(f"Type: {df[col].dtype}")
        print(f"Number of unique values: {df[col].nunique()}")

        uniques = df[col].dropna().unique()

        # Print unique values
        if len(uniques) <= MAX_UNIQUE_TO_PRINT:
            print("Values:")
            print(uniques)
        # else:
        #     print(f"Showing first {MAX_UNIQUE_TO_PRINT} values:")
        #     print(uniques[:MAX_UNIQUE_TO_PRINT])

        # Missing values?
        if df[col].isna().any():
            print(f"⚠️  Missing values: {df[col].isna().sum()}")

    print("\n")


def main():
    for file in FILES:
        path = os.path.join(DATA_DIR, file)
        if not os.path.exists(path):
            print(f"⚠️  File missing: {file}")
            continue

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"❌ Error reading {file}: {e}")
            continue
        
        inspect_dataframe(file, df)


if __name__ == "__main__":
    main()
