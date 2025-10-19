import pandas as pd
import itertools
import numpy as np

def find_top_interactions(df, target_col="accident_risk", degree=4, top_n=20):
    """
    Automatically generate feature interactions (up to given degree)
    and find the ones most correlated with the target variable.
    Excludes first-degree (original) features.
    """

    # --- 1️⃣ Encode categorical features ---
    df_encoded = df.copy()
    cat_cols = df_encoded.select_dtypes(include=['object', 'bool', 'category']).columns

    # Encode categorical columns numerically (for correlation)
    for col in cat_cols:
        df_encoded[col] = df_encoded[col].astype('category').cat.codes

    # --- 2️⃣ Identify numeric columns ---
    numeric_cols = [col for col in df_encoded.columns if col != target_col]

    # --- 3️⃣ Generate interactions ---
    interaction_features = {}
    for r in range(2, degree + 1):  # degree 2 and 3
        for combo in itertools.combinations(numeric_cols, r):
            new_name = "_x_".join(combo)
            try:
                # Multiply all features in combo
                interaction_features[new_name] = df_encoded.loc[:, combo].prod(axis=1)
            except Exception:
                continue

    # --- 4️⃣ Combine into a new DataFrame ---
    interaction_df = pd.DataFrame(interaction_features)

    # --- 5️⃣ Compute correlations with target ---
    all_corr = {}
    for col in interaction_df.columns:
        corr_val = np.corrcoef(df_encoded[target_col], interaction_df[col])[0, 1]
        if not np.isnan(corr_val):
            all_corr[col] = corr_val

    # --- 6️⃣ Sort and select top correlations ---
    top_features = sorted(all_corr.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    top_df = pd.DataFrame(top_features, columns=["feature", "correlation"])

    print(f"✅ Found {len(interaction_features)} new interaction features.")
    print(f"🔥 Top {top_n} correlated interaction features with '{target_col}':")
    print(top_df)

    return top_df, interaction_df[top_df["feature"].values]

train_df = pd.read_csv("/home/awail/PycharmProjects/kaggle/RoadAccident/train.csv")

# Drop ID if present
if "id" in train_df.columns:
    train_df = train_df.drop("id", axis=1)

top_corr_features, top_interactions_df = find_top_interactions(train_df, target_col="accident_risk", degree=3, top_n=20)
