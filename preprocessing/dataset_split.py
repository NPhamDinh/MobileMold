import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load metadata
df = pd.read_csv('metadata.csv')

df_copy = df.copy()

# Shuffle the DataFrame
df_copy = df_copy.sample(frac=1).reset_index(drop=True)

# 1. Randomly reduce 'toast' with mold=False to exactly 513 samples
toast_no_mold = df_copy[(df_copy['food'] == 'toast') & (df_copy['mold'] == False)]
if len(toast_no_mold) > 513:
    drop_indices = np.random.choice(toast_no_mold.index, len(toast_no_mold)-513, replace=False)
    df_copy = df_copy.drop(drop_indices)

# Calculate exact 10% sizes
total_samples = len(df_copy)
test_val_size = int(total_samples * 0.1)  # Exactly 10%

# 2. Create balanced test set (mold balanced, food distribution preserved)
test_dfs = []
for food in df_copy['food'].unique():
    food_subset = df_copy[df_copy['food'] == food]
    food_ratio = len(food_subset)/total_samples
    food_test_size = max(2, round(test_val_size * food_ratio))  # At least 1 per mold group
    
    # Get balanced mold samples for this food
    mold_true = food_subset[food_subset['mold'] == True]
    mold_false = food_subset[food_subset['mold'] == False]
    samples_per_group = min(len(mold_true), len(mold_false), food_test_size//2)
    
    if samples_per_group > 0:
        test_dfs.append(mold_true.sample(samples_per_group, random_state=42))
        test_dfs.append(mold_false.sample(samples_per_group, random_state=42))

test_df = pd.concat(test_dfs)
remaining_df = df_copy[~df_copy.index.isin(test_df.index)]

# 3. Create validation set (same food distribution, same size as test)
val_dfs = []
for food in df_copy['food'].unique():
    food_subset = remaining_df[remaining_df['food'] == food]
    food_ratio = len(food_subset)/len(remaining_df)
    food_val_size = max(1, round(test_val_size * food_ratio))
    
    val_dfs.append(food_subset.sample(food_val_size, random_state=42))

val_df = pd.concat(val_dfs)
remaining_df = remaining_df[~remaining_df.index.isin(val_df.index)]

# 4. Ensure exact size match by adjusting
size_diff = len(test_df) - len(val_df)
if size_diff > 0:
    # Add samples to val to match test size
    extra_samples = remaining_df.sample(size_diff, random_state=42, 
                                     weights=remaining_df['food'].value_counts(normalize=True))
    val_df = pd.concat([val_df, extra_samples])
    remaining_df = remaining_df[~remaining_df.index.isin(extra_samples.index)]
elif size_diff < 0:
    # Remove samples from val to match test size
    val_df = val_df.sample(len(test_df), random_state=42)

# 5. Train set gets everything else
train_df = remaining_df

# Verify counts and distributions
print(f"\nTotal samples: {total_samples}")
print(f"Train: {len(train_df)} ({len(train_df)/total_samples:.1%})")
print(f"Val: {len(val_df)} ({len(val_df)/total_samples:.1%})")
print(f"Test: {len(test_df)} ({len(test_df)/total_samples:.1%})")

print("\nTest set mold balance:")
print(test_df['mold'].value_counts())

print("\nFood distribution comparison:")
print(pd.DataFrame({
    'Original': df_copy['food'].value_counts(normalize=True),
    'Test': test_df['food'].value_counts(normalize=True),
    'Val': val_df['food'].value_counts(normalize=True)
}))

print("\nToast samples with mold=False:")
print(f"Final count: {len(df_copy[(df_copy['food'] == 'toast') & (df_copy['mold'] == False)])}")

# Save splits
train_df.to_csv('train_metadata.csv', index=False)
val_df.to_csv('val_metadata.csv', index=False)
test_df.to_csv('test_metadata.csv', index=False)