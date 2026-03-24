import pandas as pd
import os
import glob

# =========================
# 📁 SET YOUR FOLDER PATH
# =========================
folder_path = r"D:\Mini Project Code\Support python code\Dataset\separated dataset"

# =========================
# 📥 READ ALL FILES
# =========================
all_files = []

# 🔥 Sort files to maintain order
csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
excel_files = sorted(glob.glob(os.path.join(folder_path, "*.xlsx")) + 
                     glob.glob(os.path.join(folder_path, "*.xls")))

# =========================
# 🔄 LOAD CSV FILES
# =========================
for file in csv_files:
    try:
        df = pd.read_csv(file)
        print(f"✅ Loaded CSV: {file} | Shape: {df.shape}")
        all_files.append(df)
    except Exception as e:
        print(f"❌ Error loading CSV {file}: {e}")

# =========================
# 🔄 LOAD EXCEL / FAKE EXCEL
# =========================
for file in excel_files:
    try:
        # Try real Excel
        df = pd.read_excel(file, engine="openpyxl")
        print(f"✅ Loaded Excel: {file} | Shape: {df.shape}")

    except Exception:
        try:
            # Fallback → CSV
            df = pd.read_csv(file)
            print(f"⚠️ Loaded as CSV (fake Excel): {file} | Shape: {df.shape}")

        except Exception as e:
            print(f"❌ Failed to load {file}: {e}")
            continue

    all_files.append(df)

# =========================
# 🔗 COMBINE (APPEND ONLY)
# =========================
if len(all_files) == 0:
    print("❌ No files found!")
    exit()

# 🔥 Append one after another (no shuffle)
combined_df = pd.concat(all_files, ignore_index=True, join='inner')

print("\n✅ Combined Shape:", combined_df.shape)
print("✅ Data appended in order (NO shuffle)")

# =========================
# 📊 LABEL DISTRIBUTION
# =========================
if 'label' in combined_df.columns:
    print("\n📊 Label Distribution:")
    print(combined_df['label'].value_counts())
else:
    print("\n⚠️ No 'label' column found!")

# =========================
# 💾 SAVE FINAL DATASET
# =========================
output_path = os.path.join(folder_path, "unshuffle_merged_dataset.csv")
combined_df.to_csv(output_path, index=False)

print(f"\n🎯 Final dataset saved at: {output_path}")