from __future__ import annotations

from pathlib import Path
import importlib
import re
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parent

TARGET_NAME_PRIORITY = [
    "target",
    "label",
    "class",
    "kelas",
    "y",
    "output",
    "result",
    "pass_fail",
    "grade_category",
    "outcome",
    "survived",
    "diagnosis",
    "species",
    "quality",
    "price",
    "salary",
    "score",
    "final_exam_score",
    "umur_tahun",
]

REQUIRED_PACKAGES = {
    "nbformat": "nbformat",
    "pandas": "pandas",
    "numpy": "numpy",
    "seaborn": "seaborn",
    "matplotlib": "matplotlib",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "openpyxl": "openpyxl",
}


def ensure_required_packages() -> None:
    missing = []
    for module_name, package_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(package_name)

    if not missing:
        return

    print("Library belum lengkap. Menginstall dependency yang diperlukan...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


ensure_required_packages()

import nbformat as nbf
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "dataset"


def load_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Format data belum didukung. Gunakan file .csv, .xlsx, atau .xls")


def is_id_like(series: pd.Series, name: str) -> bool:
    lname = name.lower()
    if "id" in lname:
        return True
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    ratio = non_null.nunique() / len(non_null)
    return ratio > 0.98 and not is_numeric_dtype(series)


def candidate_priority(name: str) -> int:
    lname = name.lower()
    for idx, token in enumerate(TARGET_NAME_PRIORITY):
        if token == lname:
            return idx
    for idx, token in enumerate(TARGET_NAME_PRIORITY):
        if token in lname:
            return idx + 100
    return 10_000


def infer_task_and_target(df: pd.DataFrame, target_override: str | None = None) -> tuple[str, str]:
    if target_override:
        if target_override not in df.columns:
            raise ValueError(f"Kolom target '{target_override}' tidak ditemukan di dataset.")
        target = target_override
    else:
        candidates = []
        n_rows = len(df)
        for col in df.columns:
            s = df[col]
            nunique = s.nunique(dropna=True)
            if nunique <= 1:
                continue
            if is_id_like(s, col):
                continue

            score = candidate_priority(col)
            if score < 10_000:
                candidates.append((score, col))

        if candidates:
            target = sorted(candidates)[0][1]
        else:
            non_id_cols = [c for c in df.columns if not is_id_like(df[c], c) and df[c].nunique(dropna=True) > 1]
            if not non_id_cols:
                raise ValueError("Tidak menemukan kolom target yang layak.")
            target = non_id_cols[-1]

    n_rows = len(df)
    series = df[target]
    nunique = series.nunique(dropna=True)
    converted = None

    if series.dtype == "object":
        converted = pd.to_numeric(series, errors="coerce")
        if converted.notna().sum() >= max(3, int(len(series) * 0.8)):
            series = converted
            nunique = series.nunique(dropna=True)

    if is_bool_dtype(series):
        task = "classification"
    elif original_is_object_like(df[target]):
        if converted is not None and is_numeric_dtype(series):
            unique_ratio = nunique / max(n_rows, 1)
            if nunique <= 10 or unique_ratio <= 0.02:
                task = "classification"
            else:
                task = "regression"
        else:
            task = "classification"
    elif series.dtype == "object":
        task = "classification"
    elif is_numeric_dtype(series):
        unique_ratio = nunique / max(n_rows, 1)
        if nunique <= 10 or unique_ratio <= 0.02:
            task = "classification"
        else:
            task = "regression"
    else:
        task = "classification"

    return task, target


def original_is_object_like(series: pd.Series) -> bool:
    return series.dtype == "object" or str(series.dtype).startswith("string")


def resolve_output_dir(output_dir_raw: str | None, dataset_path: Path, task: str) -> Path:
    if output_dir_raw:
        output_dir = Path(output_dir_raw)
        if not output_dir.is_absolute():
            output_dir = (BASE_DIR / output_dir).resolve()
        return output_dir
    return BASE_DIR


def notebook_to_python_script(nb: nbf.NotebookNode) -> str:
    parts: list[str] = [
        "# Auto-generated from notebook output.",
        "# This script mirrors the generated .ipynb content.",
    ]

    for cell in nb.cells:
        if cell.cell_type == "markdown":
            markdown_lines = cell.source.strip().splitlines()
            if markdown_lines:
                parts.append("")
                parts.extend(f"# {line}" if line else "#" for line in markdown_lines)
        elif cell.cell_type == "code":
            code = cell.source.rstrip()
            parts.append("")
            if code:
                parts.append(code)
            else:
                parts.append("pass")

    return "\n".join(parts).rstrip() + "\n"


def write_python_script(nb: nbf.NotebookNode, output_path: Path) -> Path:
    py_output_path = output_path.with_suffix(".py")
    py_output_path.write_text(notebook_to_python_script(nb), encoding="utf-8")
    return py_output_path


def build_common_setup_cell(dataset_path: Path) -> str:
    rel_path = dataset_path.as_posix()
    return f"""from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import display
from scipy.stats import chi2_contingency, f_oneway, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

plt.style.use("seaborn-v0_8")
pd.set_option("display.max_columns", 100)
pd.set_option("display.float_format", lambda x: f"{{x:,.3f}}")

data_path = Path(r"{rel_path}")

if data_path.suffix.lower() == ".csv":
    df_raw = pd.read_csv(data_path)
else:
    df_raw = pd.read_excel(data_path)

print(f"Dataset path : {{data_path.resolve()}}")
print(f"Shape awal   : {{df_raw.shape}}")
"""


def build_shared_preprocessing_cell(target: str) -> str:
    return f"""df = df_raw.copy()

target_col = "{target}"

for col in df.columns:
    if df[col].dtype == "object":
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() >= max(3, int(len(df) * 0.8)):
            df[col] = converted

print("Missing value per kolom sebelum cleaning:")
display(df.isna().sum().sort_values(ascending=False).to_frame("missing_count"))
print(f"Jumlah duplikat sebelum cleaning: {{df.duplicated().sum()}}")

df = df.dropna(subset=[target_col]).drop_duplicates().reset_index(drop=True)

feature_cols = [col for col in df.columns if col != target_col]
constant_cols = [col for col in feature_cols if df[col].nunique(dropna=True) <= 1]
feature_cols = [col for col in feature_cols if col not in constant_cols]

id_like_cols = []
for col in feature_cols:
    lname = col.lower()
    if "id" in lname:
        id_like_cols.append(col)
        continue
    series = df[col].dropna()
    if len(series) == 0:
        continue
    ratio = series.nunique() / len(series)
    if ratio > 0.98 and df[col].dtype == "object":
        id_like_cols.append(col)

feature_cols = [col for col in feature_cols if col not in id_like_cols]

print(f"Shape setelah cleaning: {{df.shape}}")
print("Fitur konstan yang dikeluarkan:", constant_cols)
print("Fitur id-like yang dikeluarkan:", id_like_cols)
print("Fitur kandidat model:", feature_cols)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
"""


def build_classification_notebook(dataset_path: Path, target: str, output_path: Path) -> nbf.NotebookNode:
    title = dataset_path.stem.replace("_", " ").replace("-", " ").title()

    cells = [
        nbf.v4.new_markdown_cell(
            f"""# Latihan Gabungan Tugas 1-7: Klasifikasi {title}

Notebook ini dibuat otomatis dari dataset `{dataset_path.name}` dengan target klasifikasi **`{target}`**.

Materi yang digabungkan:

- Tugas 1: deskripsi data
- Tugas 2: data preprocessing
- Tugas 3: data transformation
- Tugas 4: imbalanced data
- Tugas 6: seleksi dan ekstraksi fitur
- Tugas 7: klasifikasi Decision Tree
"""
        ),
        nbf.v4.new_code_cell(build_common_setup_cell(dataset_path)),
        nbf.v4.new_markdown_cell("## 1. Deskripsi Data"),
        nbf.v4.new_code_cell(
            """display(df_raw.head())
print("\\nInformasi dataset awal")
df_raw.info()

print("\\nStatistik deskriptif numerik")
display(df_raw.describe(include=[np.number]).T)

print("\\nStatistik deskriptif kategorikal")
display(df_raw.describe(include=["object", "category", "bool"]).T)
"""
        ),
        nbf.v4.new_markdown_cell("## 2. Data Preprocessing"),
        nbf.v4.new_code_cell(build_shared_preprocessing_cell(target)),
        nbf.v4.new_code_cell(
            f"""numeric_feature_candidates = [col for col in feature_cols if col in numeric_cols]

def iqr_outlier_summary(frame, columns):
    rows = []
    for col in columns:
        q1 = frame[col].quantile(0.25)
        q3 = frame[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = ((frame[col] < lower) | (frame[col] > upper)).sum()
        rows.append({{
            "column": col,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": int(outlier_count),
        }})
    return pd.DataFrame(rows).sort_values("outlier_count", ascending=False)


def cap_outliers_iqr(frame, columns):
    capped = frame.copy()
    for col in columns:
        q1 = capped[col].quantile(0.25)
        q3 = capped[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        capped[col] = capped[col].clip(lower, upper)
    return capped


outlier_summary_before = iqr_outlier_summary(df, numeric_feature_candidates) if numeric_feature_candidates else pd.DataFrame()
display(outlier_summary_before)

df_clean = cap_outliers_iqr(df, numeric_feature_candidates) if numeric_feature_candidates else df.copy()
"""
        ),
        nbf.v4.new_markdown_cell("## 3. Data Transformation"),
        nbf.v4.new_code_cell(
            """numeric_model_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df_clean[col])]
transform_base = df_clean[numeric_model_cols].copy()

if len(transform_base.columns) > 0:
    simple_scaled = transform_base.divide(transform_base.max().replace(0, 1))
    minmax_scaled = (transform_base - transform_base.min()) / (transform_base.max() - transform_base.min()).replace(0, 1)
    zscore_scaled = (transform_base - transform_base.mean()) / transform_base.std().replace(0, 1)

    comparison = pd.DataFrame({
        "original_mean": transform_base.mean(),
        "simple_mean": simple_scaled.mean(),
        "minmax_mean": minmax_scaled.mean(),
        "zscore_mean": zscore_scaled.mean(),
    })
    display(comparison.round(3))
else:
    print("Tidak ada fitur numerik untuk transformasi.")
"""
        ),
        nbf.v4.new_markdown_cell("## 4. Imbalanced Data"),
        nbf.v4.new_code_cell(
            f"""target_counts = df_clean["{target}"].value_counts(dropna=False)
imbalance_ratio = target_counts.max() / target_counts.min()

display(pd.DataFrame({{"count": target_counts, "percentage": (target_counts / len(df_clean) * 100).round(2)}}))

plt.figure(figsize=(8, 4))
sns.barplot(x=target_counts.index.astype(str), y=target_counts.values, palette="Set2")
plt.title(f"Distribusi {target} | ratio={{imbalance_ratio:.2f}}")
plt.xlabel("{target}")
plt.ylabel("Jumlah data")
plt.xticks(rotation=30)
plt.show()

if imbalance_ratio < 1.5:
    print("Kesimpulan: target relatif seimbang.")
elif imbalance_ratio < 3:
    print("Kesimpulan: target moderately imbalanced.")
else:
    print("Kesimpulan: target highly imbalanced.")
"""
        ),
        nbf.v4.new_markdown_cell("## 5. Seleksi Fitur"),
        nbf.v4.new_code_cell(
            f"""analysis_df = df_clean[feature_cols + ["{target}"]].copy()
categorical_features = [col for col in feature_cols if not pd.api.types.is_numeric_dtype(analysis_df[col])]
numeric_features = [col for col in feature_cols if pd.api.types.is_numeric_dtype(analysis_df[col])]

chi_square_results = []
for col in categorical_features:
    contingency = pd.crosstab(analysis_df[col], analysis_df["{target}"])
    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
        chi2, p_value, _, _ = chi2_contingency(contingency)
        chi_square_results.append({{
            "feature": col,
            "test": "chi_square",
            "statistic": chi2,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }})

anova_results = []
for col in numeric_features:
    groups = [grp[col].values for _, grp in analysis_df.groupby("{target}")]
    if len(groups) >= 2:
        f_stat, p_value = f_oneway(*groups)
        anova_results.append({{
            "feature": col,
            "test": "anova",
            "statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }})

selection_results = pd.concat(
    [pd.DataFrame(chi_square_results), pd.DataFrame(anova_results)],
    ignore_index=True,
).sort_values(["significant", "p_value"], ascending=[False, True])

display(selection_results)

selected_features = selection_results.loc[selection_results["significant"], "feature"].tolist()
if not selected_features:
    selected_features = feature_cols.copy()

print("Fitur terpilih:", selected_features)
"""
        ),
        nbf.v4.new_markdown_cell("## 6. Ekstraksi Fitur dengan PCA"),
        nbf.v4.new_code_cell(
            f"""X_selected = df_clean[selected_features].copy()
X_selected_encoded = pd.get_dummies(X_selected, drop_first=False)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected_encoded)

pca_full = PCA()
pca_full.fit(X_scaled)

explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

print(f"Jumlah fitur setelah encoding: {{X_selected_encoded.shape[1]}}")
print(f"Jumlah komponen untuk >=95% variansi: {{n_components_95}}")

pca = PCA(n_components=n_components_95)
X_pca = pca.fit_transform(X_scaled)
pca_columns = [f"PC{{i+1}}" for i in range(n_components_95)]
df_pca = pd.DataFrame(X_pca, columns=pca_columns)
df_pca["{target}"] = pd.factorize(df_clean["{target}"])[0]
display(df_pca.head())
"""
        ),
        nbf.v4.new_markdown_cell("## 7. Klasifikasi Decision Tree"),
        nbf.v4.new_code_cell(
            f"""from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree

model_df = df_clean[selected_features + ["{target}"]].copy()
X = pd.get_dummies(model_df[selected_features], drop_first=False)
y = pd.factorize(model_df["{target}"])[0]
class_names = [str(x) for x in pd.unique(model_df["{target}"])]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi: {{accuracy:.4f}}")
print("\\nClassification Report")
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Decision Tree")
plt.show()

importance_df = pd.DataFrame({{
    "feature": X.columns,
    "importance": clf.feature_importances_,
}}).sort_values("importance", ascending=False)
display(importance_df.head(15))
"""
        ),
        nbf.v4.new_code_cell(
            """plt.figure(figsize=(24, 12))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=8,
)
plt.title("Visualisasi Decision Tree")
plt.show()
"""
        ),
        nbf.v4.new_markdown_cell(
            f"""## Kesimpulan

- Notebook ini dibuat otomatis dari dataset `{dataset_path.name}`.
- Task terdeteksi sebagai **classification** dengan target **`{target}`**.
- Materi Tugas 1-7 telah digabung ke satu alur analisis dan modeling.
"""
        ),
    ]

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
        "generated_output": str(output_path),
    }
    return nb


def build_regression_notebook(dataset_path: Path, target: str, output_path: Path) -> nbf.NotebookNode:
    title = dataset_path.stem.replace("_", " ").replace("-", " ").title()

    cells = [
        nbf.v4.new_markdown_cell(
            f"""# Latihan Gabungan Tugas 1-7: Regresi {title}

Notebook ini dibuat otomatis dari dataset `{dataset_path.name}` dengan target regresi **`{target}`**.

Materi yang digabungkan:

- Tugas 1: deskripsi data
- Tugas 2: data preprocessing
- Tugas 3: data transformation
- Tugas 4: analisis distribusi target
- Tugas 6: seleksi dan ekstraksi fitur
- Tugas 7: Decision Tree Regressor
"""
        ),
        nbf.v4.new_code_cell(build_common_setup_cell(dataset_path)),
        nbf.v4.new_markdown_cell("## 1. Deskripsi Data"),
        nbf.v4.new_code_cell(
            """display(df_raw.head())
print("\\nInformasi dataset awal")
df_raw.info()

print("\\nStatistik deskriptif awal")
display(df_raw.describe(include="all").T)
"""
        ),
        nbf.v4.new_markdown_cell("## 2. Data Preprocessing"),
        nbf.v4.new_code_cell(build_shared_preprocessing_cell(target)),
        nbf.v4.new_code_cell(
            f"""if pd.api.types.is_numeric_dtype(df[target_col]):
    def iqr_outlier_summary(frame, col):
        q1 = frame[col].quantile(0.25)
        q3 = frame[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = ((frame[col] < lower) | (frame[col] > upper)).sum()
        return pd.DataFrame([{{
            "column": col,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": int(outlier_count),
        }}])

    display(iqr_outlier_summary(df, target_col))
else:
    print("Target belum numerik setelah cleaning. Dataset tidak cocok untuk regresi.")

df_clean = df.copy()
"""
        ),
        nbf.v4.new_markdown_cell("## 3. Data Transformation"),
        nbf.v4.new_code_cell(
            """numeric_feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df_clean[col])]
transform_base = df_clean[numeric_feature_cols].copy()

if len(transform_base.columns) > 0:
    simple_scaled = transform_base.divide(transform_base.max().replace(0, 1))
    minmax_scaled = (transform_base - transform_base.min()) / (transform_base.max() - transform_base.min()).replace(0, 1)
    zscore_scaled = (transform_base - transform_base.mean()) / transform_base.std().replace(0, 1)

    comparison = pd.DataFrame({
        "original_mean": transform_base.mean(),
        "simple_mean": simple_scaled.mean(),
        "minmax_mean": minmax_scaled.mean(),
        "zscore_mean": zscore_scaled.mean(),
    })
    display(comparison.round(3))
else:
    print("Tidak ada fitur numerik untuk transformasi.")
"""
        ),
        nbf.v4.new_markdown_cell("## 4. Analisis Distribusi Target"),
        nbf.v4.new_code_cell(
            f"""plt.figure(figsize=(10, 4))
sns.histplot(df_clean["{target}"], bins=25, kde=True, color="#3182bd")
plt.title("Distribusi Target")
plt.xlabel("{target}")
plt.show()

target_bins = pd.cut(df_clean["{target}"], bins=4, duplicates="drop")
bin_counts = target_bins.value_counts().sort_index()
imbalance_ratio = bin_counts.max() / bin_counts.min()
display(pd.DataFrame({{"count": bin_counts, "percentage": (bin_counts / len(df_clean) * 100).round(2)}}))
print(f"Imbalance ratio antar bin target: {{imbalance_ratio:.2f}}")
"""
        ),
        nbf.v4.new_markdown_cell("## 5. Seleksi Fitur"),
        nbf.v4.new_code_cell(
            f"""selection_base = df_clean[feature_cols].copy()

for col in selection_base.columns:
    if not pd.api.types.is_numeric_dtype(selection_base[col]):
        selection_base[col] = pd.factorize(selection_base[col])[0]

selection_results = []
for col in selection_base.columns:
    if selection_base[col].nunique() <= 1:
        continue
    pearson_corr, pearson_p = pearsonr(selection_base[col], df_clean["{target}"])
    spearman_corr, spearman_p = spearmanr(selection_base[col], df_clean["{target}"])
    selection_results.append({{
        "feature": col,
        "pearson_corr": pearson_corr,
        "pearson_p_value": pearson_p,
        "spearman_corr": spearman_corr,
        "spearman_p_value": spearman_p,
        "significant": (pearson_p < 0.05) or (spearman_p < 0.05),
    }})

selection_df = pd.DataFrame(selection_results).sort_values(
    ["significant", "spearman_p_value", "pearson_p_value"],
    ascending=[False, True, True],
)
display(selection_df.round(4))

selected_features = selection_df.loc[selection_df["significant"], "feature"].tolist()
if not selected_features:
    selected_features = feature_cols.copy()

print("Fitur terpilih:", selected_features)
"""
        ),
        nbf.v4.new_markdown_cell("## 6. Ekstraksi Fitur dengan PCA"),
        nbf.v4.new_code_cell(
            f"""X_selected = df_clean[selected_features].copy()
X_selected = pd.get_dummies(X_selected, drop_first=False)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

pca_full = PCA()
pca_full.fit(X_scaled)

explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

print(f"Jumlah fitur setelah encoding: {{X_selected.shape[1]}}")
print(f"Jumlah komponen untuk >=95% variansi: {{n_components_95}}")

pca = PCA(n_components=n_components_95)
X_pca = pca.fit_transform(X_scaled)
pca_columns = [f"PC{{i+1}}" for i in range(n_components_95)]
df_pca = pd.DataFrame(X_pca, columns=pca_columns)
df_pca["{target}"] = df_clean["{target}"].values
display(df_pca.head())
"""
        ),
        nbf.v4.new_markdown_cell("## 7. Regresi Decision Tree"),
        nbf.v4.new_code_cell(
            f"""from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree

model_df = df_clean[selected_features + ["{target}"]].copy()
X = pd.get_dummies(model_df[selected_features], drop_first=False)
y = model_df["{target}"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

reg = DecisionTreeRegressor(random_state=42, max_depth=5, min_samples_leaf=10)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"MAE  : {{mae:.4f}}")
print(f"RMSE : {{rmse:.4f}}")
print(f"R^2  : {{r2:.4f}}")

importance_df = pd.DataFrame({{
    "feature": X.columns,
    "importance": reg.feature_importances_,
}}).sort_values("importance", ascending=False)
display(importance_df.head(15))
"""
        ),
        nbf.v4.new_code_cell(
            """fig, axes = plt.subplots(1, 2, figsize=(18, 6))

axes[0].scatter(y_test, y_pred, alpha=0.6, color="#3182bd")
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
axes[0].set_xlabel("Actual")
axes[0].set_ylabel("Predicted")
axes[0].set_title("Actual vs Predicted")

residuals = y_test - y_pred
axes[1].hist(residuals, bins=25, color="#9ecae1", edgecolor="black")
axes[1].set_title("Distribusi Residual")

plt.tight_layout()
plt.show()

plt.figure(figsize=(24, 12))
plot_tree(reg, feature_names=X.columns, filled=True, rounded=True, fontsize=8)
plt.title("Visualisasi Decision Tree Regressor")
plt.show()
"""
        ),
        nbf.v4.new_markdown_cell(
            f"""## Kesimpulan

- Notebook ini dibuat otomatis dari dataset `{dataset_path.name}`.
- Task terdeteksi sebagai **regression** dengan target **`{target}`**.
- Materi Tugas 1-7 telah digabung ke satu alur analisis dan modeling.
"""
        ),
    ]

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
        "generated_output": str(output_path),
    }
    return nb


def generate_notebook_for_dataset(
    dataset_path: Path,
    output_dir_raw: str | None = None,
    target_override: str | None = None,
) -> tuple[Path, Path]:
    df = load_dataset(dataset_path)
    task, target = infer_task_and_target(df, target_override=target_override)
    output_dir = resolve_output_dir(output_dir_raw, dataset_path, task)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = slugify(dataset_path.stem)
    output_path = output_dir / f"latihan_1_7_{base_name}_{task}.ipynb"

    if task == "classification":
        nb = build_classification_notebook(dataset_path, target, output_path)
    else:
        nb = build_regression_notebook(dataset_path, target, output_path)

    nbf.write(nb, output_path)
    py_output_path = write_python_script(nb, output_path)
    return output_path, py_output_path


def main() -> int:
    raw_path = sys.argv[1] if len(sys.argv) > 1 else input("Masukkan path dataset (.csv/.xlsx): ").strip()
    output_dir_raw = sys.argv[2] if len(sys.argv) > 2 else input("Masukkan path folder output notebook: ").strip()
    target_override = sys.argv[3] if len(sys.argv) > 3 else input("Masukkan nama kolom target (boleh kosong untuk auto): ").strip()
    if not raw_path:
        print("Path dataset kosong.")
        return 1
    if not target_override:
        target_override = None
    if not output_dir_raw:
        output_dir_raw = None

    dataset_path = Path(raw_path)
    if not dataset_path.is_absolute():
        dataset_path = (BASE_DIR / dataset_path).resolve()

    if not dataset_path.exists():
        print(f"File tidak ditemukan: {dataset_path}")
        return 1

    try:
        output_path, py_output_path = generate_notebook_for_dataset(dataset_path, output_dir_raw, target_override)
    except Exception as exc:
        print(f"Gagal membuat notebook: {exc}")
        return 1

    print(f"Notebook berhasil dibuat: {output_path}")
    print(f"Script Python berhasil dibuat: {py_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
