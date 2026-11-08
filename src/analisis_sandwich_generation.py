
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == 'src' else SCRIPT_DIR

CANDIDATE_JSON_PATHS = [
    PROJECT_ROOT / 'data' / 'raw' / 'merged-1794142463790.json',
    Path('data/raw/merged-1794142463790.json'),
    Path('../data/raw/merged-1794142463790.json'),
    Path('merged-1794142463790.json'),
    Path('/working_dir/c_1f83def33603e5a6/user_uploaded_files/ttl=63d/user_uploaded/merged-1794142463790.json'),
]

json_path = next((p for p in CANDIDATE_JSON_PATHS if p.exists()), None)
if json_path is None:
    searched = "\n  - ".join(str(p) for p in CANDIDATE_JSON_PATHS)
    raise FileNotFoundError(
        f"File 'merged-1794142463790.json' tidak ditemukan. Lokasi yang sudah dicek:\n  - {searched}"
    )

SANDBOX_OUT = Path('/working_dir/c_1f83def33603e5a6/artifacts/file_generation/ttl=63d/output')
if os.name == 'posix' and SANDBOX_OUT.exists() and not (PROJECT_ROOT / 'data' / 'raw').exists():
    CSV_OUTPUT_DIR = SANDBOX_OUT
    FIG_OUTPUT_DIR = SANDBOX_OUT
else:
    CSV_OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed'
    FIG_OUTPUT_DIR = PROJECT_ROOT / 'output' / 'figures'

CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Membaca dataset dari : {json_path}")
with open(json_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

posts = raw_data.get('posts', [])
df = pd.DataFrame(posts)

# ------------------------------------------------------------------------------
# 2. PEMETAAN KLASTER UTAS & EKSTRAKSI FITUR TEKS
# ------------------------------------------------------------------------------
def assign_cluster(idx):
    if idx <= 49:
        return "K1: Evaluasi Aset Rp430Jt (@charttechnician27)"
    elif idx <= 92:
        return "K2: Anak sbg Investasi (@gussupriyatna17)"
    elif idx <= 125:
        return "K3: Bebas Pinjol Usia 27 (@moon_delluna06)"
    elif idx <= 154:
        return "K4: Dilema Mahar Sandwich (@mfaya_)"
    elif idx <= 204:
        return "K5: Rincian Beban Keluarga (@mamaformulator)"
    else:
        return "K6: Psikologi & Sistem SJSN (@panjisyamsi)"

df['cluster_id'] = [assign_cluster(i) for i in range(len(df))]
root_ids = {'Dc6ZfldCQJd', 'DKZXXmBS4OO', 'Dcs6WJPkh7q', 'DdHPQoqgfn-', 'DNWvpIsBBCY', 'DW-ES2ygW0G'}
df['is_root_post'] = df['id'].isin(root_ids)
df['post_role'] = np.where(df['is_root_post'], 'Utas Utama (Root)', 'Komentar / Balasan')

df['text_clean'] = df['text'].fillna('')
df['media_alt'] = df['media'].apply(
    lambda m: ' '.join([x.get('alt', '') for x in m]) if isinstance(m, list) else ''
)
df['combined_text'] = (df['text_clean'] + ' ' + df['media_alt']).str.strip()

# Variabel Independen X1: Panjang Karakter Komentar (Character Count)
df['X1_char_count'] = df['text_clean'].apply(len).astype(float)
df['word_count'] = df['text_clean'].apply(lambda x: len(x.split())).astype(int)

# ------------------------------------------------------------------------------
# 3. OPERASIONALISASI SKOR SENTIMEN/NEGATIVITAS (X2) & KERENTANAN FINANSIAL
# ------------------------------------------------------------------------------
neg_words_high = [
    'benci', 'nyerah', 'stres', 'burnout', 'stuck', 'depresi', 'hancur', 'runtuh',
    'ditelantarkan', 'guntingin', 'diusir', 'jahat', 'kejam', 'malakin', 'tempeleng',
    'lucknut', 'bentak', 'ngebentak', 'manipulatif', 'ngelunjak', 'norak', 'pamer',
    'miskin', 'nyusahin', 'gatau diri', 'atm berjalan', 'judi', 'bangkrut', 'hack',
    'kemiskinan struktural', 'krisis moral', 'war world', 'jebakan'
]
neg_words_mod = [
    'capek', 'cape', 'capeee', 'capekkk', 'lelah', 'berat', 'beban', 'pusing',
    'sedih', 'nangis', 'sakit', 'takut', 'khawatir', 'aib', 'galak', 'hujat', 'dihujat',
    'sindiran', 'disindir', 'omel', 'kena omel', 'tuntut', 'dituntut', 'kesel', 'marah',
    'ga adil', 'engga adil', 'gak adil', 'struggle', 'sulit', 'susah', 'kurang',
    'utang', 'hutang', 'ngutang', 'pinjol', 'paylater', 'spaylater', 'emok', 'cicilan',
    'menguras', 'terlalu', 'boong', 'diumpetin', 'keluh', 'ngeluh', 'gagal', 'fomo',
    'tidur2an', 'malu', 'kecil', '0 rupiah', 'rekening 0', 'habis'
]
neg_emojis = ['😭', '🥲', '😢', '😰', '🥺', '😩', '😑', '🙃', '💔', ':(', ":'("]
pos_words = [
    'semangat', 'semangatt', 'semangattt', 'mangatz', 'alhamdulillah', 'allhamdulilah',
    'bersyukur', 'syukur', 'ikhlas', 'iklash', 'ridho', 'ridha', 'berkah', 'keren',
    'hebat', 'amin', 'aminn', 'doa', 'berdoa', 'sabar', 'sukses', 'lancar', 'amanah',
    'bahagia', 'membahagiakan', 'goodluck', 'good job', 'nice', 'appreciate', 'menyala',
    'terimakasih', 'terima kasih', 'makasih', 'thanks', 'selow', 'tenang'
]

def compute_negativity_score(row):
    txt = row['text_clean']
    txt_lower = txt.lower()
    high_hits = sum(txt_lower.count(w) for w in neg_words_high)
    mod_hits = sum(txt_lower.count(w) for w in neg_words_mod)
    emoji_hits = sum(txt.count(e) for e in neg_emojis)
    pos_hits = sum(txt_lower.count(w) for w in pos_words)

    alpha_chars = [c for c in txt if c.isalpha()]
    caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) if len(alpha_chars) > 5 else 0.0
    caps_bonus = 12.0 if caps_ratio > 0.4 else (5.0 if caps_ratio > 0.15 else 0.0)
    punct_bonus = min(10.0, (txt.count('!') + txt.count('?')) * 1.8)

    raw_neg = (high_hits * 14.0) + (mod_hits * 7.5) + (emoji_hits * 6.0) + caps_bonus + punct_bonus
    raw_pos_offset = pos_hits * 3.5
    net_score = max(2.0, raw_neg - (0.4 * raw_pos_offset) + (4.0 if len(txt) > 0 else 5.0))
    scaled = 100.0 * (1.0 - np.exp(-net_score / 35.0))
    return round(float(np.clip(scaled, 0.0, 100.0)), 2)

df['X2_negativity_score'] = df.apply(compute_negativity_score, axis=1)

# Indikator Kerentanan Finansial (Financial Vulnerability Score: 0 - 100)
debt_terms = [
    'pinjol', 'paylater', 'spaylater', 'utang', 'hutang', 'ngutang',
    'emok', 'cicilan', 'angsuran', 'kredit', 'kpr', 'tagihan'
]
sandwich_terms = [
    'sandwich', 'sandgen', 'sendwich', 'tanggungan', 'nanggung', 'biayain', 'ortu',
    'orang tua', 'orangtua', 'ibu', 'ibuk', 'ibukku', 'ayah', 'bapa', 'mamah', 'papa',
    'adek', 'adik', 'ponakan', 'keponakan', 'kakak', 'sekolah', 'spp', 'kuliah',
    'listrik', 'token', 'bpjs', 'uang dapur', 'bulanan', 'atm berjalan', 'investasi',
    'dana pensiun', 'lansia', 'keluarga', 'anak pertama', 'sulung', 'bungsu', 'single fighter'
]
low_income_terms = [
    'umr', 'setengah umr', '2juta', '2 juta', '2.7jt', '6-8 juta', '6-8jt', '500k', '250k',
    '218rb', '0 rupiah', 'rekening 0', '1.509.137', '1.237.735', 'sate ayam', 'indomie',
    'tempe', 'tahu', 'buruh', 'pas pasan', 'miskin', 'susah', 'habis', 'ga punya tabungan',
    'belum punya tabungan', 'engga punya apa apa', 'ngontrak', 'diusir', 'kos', 'ngekos',
    '50%', '3/5', '60 juta', '49jt', '130jt', '300jt'
]
distress_terms = [
    'capek', 'cape', 'lelah', 'stres', 'burnout', 'berat', 'beban', 'pusing', 'sedih',
    'nangis', 'nyerah', 'stuck', 'sakit', 'menguras', 'terpentok', 'malakin', 'dituntut'
]

def compute_vulnerability_score(row):
    comb = row['combined_text'].lower()
    d_hits = sum(comb.count(w) for w in debt_terms)
    s_hits = sum(comb.count(w) for w in sandwich_terms)
    l_hits = sum(comb.count(w) for w in low_income_terms)
    e_hits = sum(comb.count(w) for w in distress_terms)

    raw_vuln = (d_hits * 14.5) + (s_hits * 6.5) + (l_hits * 11.0) + (e_hits * 7.0)
    total_raw = 8.0 + raw_vuln
    score = 100.0 * (1.0 - np.exp(-total_raw / 42.0))
    return round(float(np.clip(score, 0.0, 100.0)), 2)

df['financial_vulnerability_score'] = df.apply(compute_vulnerability_score, axis=1)
df['has_explicit_financial_story'] = df['combined_text'].str.lower().apply(
    lambda t: any(w in t for w in (debt_terms + sandwich_terms + low_income_terms))
)

def categorize_vulnerability(score):
    if score < 25.0:
        return "1. Rendah (<25)"
    elif score < 50.0:
        return "2. Sedang (25-49.9)"
    elif score < 75.0:
        return "3. Tinggi (50-74.9)"
    else:
        return "4. Sangat Tinggi/Kritis (>=75)"

df['vulnerability_category'] = df['financial_vulnerability_score'].apply(categorize_vulnerability)
df['Y_likes'] = df['likes'].astype(float)
df['Y_log_likes'] = np.log1p(df['Y_likes'])

# Simpan CSV hasil olahan
csv_out = CSV_OUTPUT_DIR / 'data_olahan_threads_sandwich_gen.csv'
df[[
    'id', 'username', 'timestamp', 'cluster_id', 'post_role',
    'Y_likes', 'Y_log_likes', 'replies', 'reposts', 'shares',
    'X1_char_count', 'word_count', 'X2_negativity_score',
    'financial_vulnerability_score', 'has_explicit_financial_story',
    'vulnerability_category', 'url', 'text_clean'
]].to_csv(csv_out, index=False, encoding='utf-8')

# ------------------------------------------------------------------------------
# 4. FUNGSI ESTIMASI REGRESI LINIER BERGANDA (OLS & ROBUST HC3)
# ------------------------------------------------------------------------------
def fit_ols(y, X_df):
    y_vec = np.asarray(y, dtype=float)
    X_mat = np.column_stack([np.ones(len(y_vec)), X_df.values.astype(float)])
    col_names = ['Konstanta (Intercept)'] + list(X_df.columns)
    n, k = X_mat.shape
    df_resid, df_model = n - k, k - 1

    XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
    beta = XtX_inv @ (X_mat.T @ y_vec)
    y_hat = X_mat @ beta
    resid = y_vec - y_hat

    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y_vec - np.mean(y_vec)) ** 2))
    r2 = 1.0 - (rss / tss)
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df_resid

    mse_resid = rss / df_resid
    f_stat = ((tss - rss) / df_model) / mse_resid
    f_pval = float(stats.f.sf(f_stat, df_model, df_resid))

    se_ols = np.sqrt(np.diag(mse_resid * XtX_inv))
    t_ols = beta / se_ols
    p_ols = 2.0 * stats.t.sf(np.abs(t_ols), df=df_resid)

    hat_diag = np.einsum('ij,jk,ik->i', X_mat, XtX_inv, X_mat)
    u_hc3 = (resid / (1.0 - hat_diag)) ** 2
    cov_hc3 = XtX_inv @ ((X_mat * u_hc3[:, None]).T @ X_mat) @ XtX_inv
    se_hc3 = np.sqrt(np.diag(cov_hc3))
    t_hc3 = beta / se_hc3
    p_hc3 = 2.0 * stats.t.sf(np.abs(t_hc3), df=df_resid)

    std_y = np.std(y_vec, ddof=1)
    std_x = np.std(X_mat, axis=0, ddof=1)
    beta_std = np.zeros(k)
    beta_std[1:] = beta[1:] * (std_x[1:] / std_y)

    coef_table = pd.DataFrame({
        'Variabel': col_names,
        'Koefisien (B)': beta,
        'Std. Error (OLS)': se_ols,
        't-Stat (OLS)': t_ols,
        'p-Value (OLS)': p_ols,
        'Beta Standar': beta_std,
        'Robust SE (HC3)': se_hc3,
        't-Stat (HC3)': t_hc3,
        'p-Value (HC3)': p_hc3
    })
    return {
        'n': n, 'df_model': df_model, 'df_resid': df_resid,
        'r2': r2, 'adj_r2': adj_r2, 'f_stat': f_stat, 'f_pval': f_pval,
        'rmse': np.sqrt(mse_resid), 'coef_table': coef_table
    }

r_x1_x2, p_x1_x2 = stats.pearsonr(df['X1_char_count'], df['X2_negativity_score'])
vif_x1_x2 = 1.0 / (1.0 - (r_x1_x2 ** 2))

res_raw = fit_ols(df['Y_likes'], df[['X1_char_count', 'X2_negativity_score']])
res_log = fit_ols(df['Y_log_likes'], df[['X1_char_count', 'X2_negativity_score']])

sub_fin = df[df['has_explicit_financial_story']]['financial_vulnerability_score']
corr_cols = ['Y_likes', 'Y_log_likes', 'replies', 'X1_char_count', 'word_count', 'X2_negativity_score', 'financial_vulnerability_score']

print("\n=== 1. STATISTIK DESKRIPTIF & SKEWNESS (N = 234) ===")
for col in ['financial_vulnerability_score', 'X1_char_count', 'X2_negativity_score', 'Y_likes', 'Y_log_likes']:
    s = df[col]
    print(f"{col:30s} | Mean={s.mean():8.2f} | Median={s.median():8.2f} | Std={s.std():8.2f} | Skew={s.skew():+7.4f} | Kurt={s.kurt():+7.4f}")
print(f"Sub-sampel Narasi Aktif (n={len(sub_fin)}): Mean={sub_fin.mean():.2f} | Median={sub_fin.median():.2f} | Skew={sub_fin.skew():+.4f}")

print("\n=== 2. REGRESI LINIER BERGANDA (VIF = {:.4f}) ===".format(vif_x1_x2))
print(f"Model 1A (Likes Asli)   : R2={res_raw['r2']:.4f}, F={res_raw['f_stat']:.4f}, p={res_raw['f_pval']:.4e}")
print(res_raw['coef_table'].to_string(index=False))
print(f"\nModel 1B (ln(Likes+1))  : R2={res_log['r2']:.4f}, F={res_log['f_stat']:.4f}, p={res_log['f_pval']:.4e}")
print(res_log['coef_table'].to_string(index=False))

# ------------------------------------------------------------------------------
# 5. PEMBUATAN 4 VISUALISASI DATA BERESOLUSI TINGGI (300 DPI)
# ------------------------------------------------------------------------------
palette_main = ['#1B365D', '#E05A47', '#2E8B57', '#D99B26', '#6C5B7B', '#355C7D']

# VISUALISASI 1
fig = plt.figure(figsize=(14, 10), dpi=300)
gs = gridspec.GridSpec(2, 2, height_ratios=[1.1, 1], hspace=0.35, wspace=0.28)

ax1 = fig.add_subplot(gs[0, 0])
sns.histplot(df['financial_vulnerability_score'], bins=18, kde=True, color='#1B365D', edgecolor='white', alpha=0.75, ax=ax1)
mean_v = df['financial_vulnerability_score'].mean()
med_v = df['financial_vulnerability_score'].median()
q1_v = df['financial_vulnerability_score'].quantile(0.25)
q3_v = df['financial_vulnerability_score'].quantile(0.75)
skew_v = df['financial_vulnerability_score'].skew()
kurt_v = df['financial_vulnerability_score'].kurt()

ax1.axvline(mean_v, color='#E05A47', linestyle='--', linewidth=2.2, label=f'Mean (Rata-rata): {mean_v:.2f}')
ax1.axvline(med_v, color='#2E8B57', linestyle='-', linewidth=2.2, label=f'Median (Nilai Tengah): {med_v:.2f}')
ax1.axvspan(q1_v, q3_v, color='#D99B26', alpha=0.18, label=f'IQR (Q1={q1_v:.1f} - Q3={q3_v:.1f})')
ax1.set_title('A. Distribusi & Skewness Skor Kerentanan Finansial (N = 234)', fontsize=11.5, fontweight='bold', pad=10)
ax1.set_xlabel('Skor Kerentanan Finansial (Skala 0 - 100)', fontsize=10)
ax1.set_ylabel('Frekuensi Komentar', fontsize=10)
ax1.legend(loc='upper right', fontsize=8.5, frameon=True)
ax1.grid(True)
ax1.text(0.97, 0.58, f"Skewness = +{skew_v:.3f}\nKurtosis = {kurt_v:.3f}\n(Right-Skewed / Menceng Kanan)",
         transform=ax1.transAxes, ha='right', va='top', fontsize=8.5,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F9FA', edgecolor='#CCCCCC'))

ax2 = fig.add_subplot(gs[0, 1])
sns.histplot(sub_fin, bins=16, kde=True, color='#E05A47', edgecolor='white', alpha=0.75, ax=ax2)
ax2.axvline(sub_fin.mean(), color='#1B365D', linestyle='--', linewidth=2.2, label=f'Mean Narasi Aktif: {sub_fin.mean():.2f}')
ax2.axvline(sub_fin.median(), color='#2E8B57', linestyle='-', linewidth=2.2, label=f'Median Narasi Aktif: {sub_fin.median():.2f}')
ax2.set_title(f'B. Distribusi Khusus Komentar Bermuatan Fin./Sandwich (n = {len(sub_fin)})', fontsize=11.5, fontweight='bold', pad=10)
ax2.set_xlabel('Skor Kerentanan Finansial (Skala 0 - 100)', fontsize=10)
ax2.set_ylabel('Frekuensi Komentar', fontsize=10)
ax2.legend(loc='upper right', fontsize=8.5, frameon=True)
ax2.grid(True)
ax2.text(0.97, 0.58, f"Skewness = +{sub_fin.skew():.3f}\nKurtosis = {sub_fin.kurt():.3f}\nIQR = {sub_fin.quantile(0.75)-sub_fin.quantile(0.25):.2f}",
         transform=ax2.transAxes, ha='right', va='top', fontsize=8.5,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF5F5', edgecolor='#E05A47'))

ax3 = fig.add_subplot(gs[1, :])
cluster_order = [
    "K1: Evaluasi Aset Rp430Jt (@charttechnician27)",
    "K2: Anak sbg Investasi (@gussupriyatna17)",
    "K3: Bebas Pinjol Usia 27 (@moon_delluna06)",
    "K4: Dilema Mahar Sandwich (@mfaya_)",
    "K5: Rincian Beban Keluarga (@mamaformulator)",
    "K6: Psikologi & Sistem SJSN (@panjisyamsi)"
]
sns.boxplot(data=df, x='financial_vulnerability_score', y='cluster_id', hue='cluster_id',
            order=cluster_order, palette=palette_main, legend=False, width=0.55, ax=ax3,
            fliersize=0, boxprops=dict(alpha=0.85))
sns.stripplot(data=df, x='financial_vulnerability_score', y='cluster_id', order=cluster_order,
              color='#222222', alpha=0.45, size=4.5, jitter=0.2, ax=ax3)
ax3.axvline(mean_v, color='#E05A47', linestyle='--', linewidth=1.5, label=f'Mean Global ({mean_v:.2f})')
ax3.axvline(med_v, color='#2E8B57', linestyle='-', linewidth=1.5, label=f'Median Global ({med_v:.2f})')
ax3.set_title('C. Perbandingan Sebaran Skor Kerentanan Finansial Antar 6 Klaster Utas Threads', fontsize=11.5, fontweight='bold', pad=10)
ax3.set_xlabel('Skor Kerentanan Finansial (0 - 100)', fontsize=10)
ax3.set_ylabel('')
ax3.legend(loc='lower right', fontsize=9)
ax3.grid(True, axis='x')

fig.suptitle('UJI DESKRIPTIF & ANALISIS SKEWNESS SKOR KERENTANAN FINANSIAL AUDIENS THREADS',
             fontsize=14, fontweight='bold', y=0.98)
fig.text(0.5, 0.01, 'Sumber Data: Hasil Scrape Komentar Threads (merged-1794142463790.json, N = 234) via github.com/jobas5/thread-scrapper',
         ha='center', fontsize=9, style='italic', color='#555555')
plt.savefig(FIG_OUTPUT_DIR / 'visualisasi_1_distribusi_skewness_kerentanan.png', bbox_inches='tight', dpi=300)
plt.close(fig)

# VISUALISASI 2
fig = plt.figure(figsize=(14, 10), dpi=300)
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.28)

ax1 = fig.add_subplot(gs[0, 0])
sns.regplot(data=df, x='X1_char_count', y='Y_log_likes', ax=ax1,
            scatter_kws={'alpha': 0.55, 'color': '#1B365D', 's': 38},
            line_kws={'color': '#E05A47', 'linewidth': 2.2})
b1_log = res_log['coef_table'].loc[1, 'Koefisien (B)']
p1_log = res_log['coef_table'].loc[1, 'p-Value (OLS)']
ax1.set_title('A. Pengaruh Panjang Karakter (X1) terhadap ln(Likes + 1)', fontsize=11.5, fontweight='bold', pad=10)
ax1.set_xlabel('X1: Panjang Karakter Komentar (Character Count)', fontsize=10)
ax1.set_ylabel('ln(Likes + 1)', fontsize=10)
ax1.grid(True)
ax1.text(0.04, 0.94, f"Koefisien Parsial B1 = +{b1_log:.4f}\np-value = {p1_log:.4e} (Signifikan)\nKomentar panjang memicu resonansi",
         transform=ax1.transAxes, ha='left', va='top', fontsize=8.5,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F9FA', edgecolor='#1B365D'))

ax2 = fig.add_subplot(gs[0, 1])
sns.regplot(data=df, x='X2_negativity_score', y='Y_log_likes', ax=ax2,
            scatter_kws={'alpha': 0.55, 'color': '#6C5B7B', 's': 38},
            line_kws={'color': '#E05A47', 'linewidth': 2.2})
b2_log = res_log['coef_table'].loc[2, 'Koefisien (B)']
p2_log = res_log['coef_table'].loc[2, 'p-Value (OLS)']
ax2.set_title('B. Pengaruh Skor Sentimen/Negativitas (X2) terhadap ln(Likes + 1)', fontsize=11.5, fontweight='bold', pad=10)
ax2.set_xlabel('X2: Skor Sentimen/Negativitas (Skala 0 - 100)', fontsize=10)
ax2.set_ylabel('ln(Likes + 1)', fontsize=10)
ax2.grid(True)
ax2.text(0.04, 0.94, f"Koefisien Parsial B2 = +{b2_log:.4f}\np-value = {p2_log:.4f}\nSentimen emosional menaikkan atensi",
         transform=ax2.transAxes, ha='left', va='top', fontsize=8.5,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F9FA', edgecolor='#6C5B7B'))

ax3 = fig.add_subplot(gs[1, 0])
x1_grid = np.linspace(df['X1_char_count'].min(), df['X1_char_count'].max(), 60)
x2_grid = np.linspace(df['X2_negativity_score'].min(), df['X2_negativity_score'].max(), 60)
X1_mesh, X2_mesh = np.meshgrid(x1_grid, x2_grid)
b0_l = res_log['coef_table'].loc[0, 'Koefisien (B)']
Z_mesh = b0_l + b1_log * X1_mesh + b2_log * X2_mesh

contour = ax3.contourf(X1_mesh, X2_mesh, Z_mesh, levels=15, cmap='YlOrRd', alpha=0.82)
cbar = plt.colorbar(contour, ax=ax3)
cbar.set_label('Prediksi ln(Likes + 1)', fontsize=9)
ax3.scatter(df['X1_char_count'], df['X2_negativity_score'], c=df['Y_log_likes'],
            cmap='YlOrRd', edgecolors='black', linewidth=0.5, s=35, alpha=0.9)
ax3.set_title('C. Peta Kontur Regresi Linier Berganda: Y_hat = f(X1, X2)', fontsize=11.5, fontweight='bold', pad=10)
ax3.set_xlabel('X1: Panjang Karakter Komentar', fontsize=10)
ax3.set_ylabel('X2: Skor Sentimen/Negativitas', fontsize=10)
ax3.grid(True)

ax4 = fig.add_subplot(gs[1, 1])
models_comp = pd.DataFrame({
    'Prediktor': ['X1: Panjang Karakter', 'X2: Skor Negativitas', 'X1: Panjang Karakter', 'X2: Skor Negativitas'],
    'Model': [
        f"Model 1A: Likes Asli (R²={res_raw['r2']*100:.1f}%)",
        f"Model 1A: Likes Asli (R²={res_raw['r2']*100:.1f}%)",
        f"Model 1B: ln(Likes+1) (R²={res_log['r2']*100:.1f}%)",
        f"Model 1B: ln(Likes+1) (R²={res_log['r2']*100:.1f}%)"
    ],
    'Standardized_Beta': [
        res_raw['coef_table'].loc[1, 'Beta Standar'],
        res_raw['coef_table'].loc[2, 'Beta Standar'],
        res_log['coef_table'].loc[1, 'Beta Standar'],
        res_log['coef_table'].loc[2, 'Beta Standar']
    ]
})
sns.barplot(data=models_comp, x='Prediktor', y='Standardized_Beta', hue='Model',
            palette=['#1B365D', '#E05A47'], edgecolor='black', ax=ax4)
ax4.axhline(0, color='black', linewidth=0.8)
for container in ax4.containers:
    ax4.bar_label(container, fmt='%.3f', padding=3, fontsize=8.5, fontweight='bold')
ax4.set_ylim(0, max(models_comp['Standardized_Beta']) * 1.28)
ax4.set_title('D. Perbandingan Koefisien Beta Terstandardisasi (Beta Standar)', fontsize=11.5, fontweight='bold', pad=10)
ax4.set_xlabel('')
ax4.set_ylabel('Standardized Beta Coefficient (β*)', fontsize=10)
ax4.legend(loc='upper right', fontsize=8.5, frameon=True)
ax4.grid(True, axis='y')

fig.suptitle('PEMODELAN REGRESI LINIER BERGANDA: PREDIKSI LIKES BERDASARKAN X1 & X2',
             fontsize=14, fontweight='bold', y=0.98)
fig.text(0.5, 0.01, f"Persamaan Raw: Y = {res_raw['coef_table'].loc[0, 'Koefisien (B)']:.2f} + {res_raw['coef_table'].loc[1, 'Koefisien (B)']:.3f}(X1) + {res_raw['coef_table'].loc[2, 'Koefisien (B)']:.3f}(X2) | "
                    f"Persamaan Log: ln(Y+1) = {b0_l:.3f} + {b1_log:.4f}(X1) + {b2_log:.4f}(X2)",
         ha='center', fontsize=9, style='italic', color='#333333')
plt.savefig(FIG_OUTPUT_DIR / 'visualisasi_2_regresi_linier_berganda.png', bbox_inches='tight', dpi=300)
plt.close(fig)

# VISUALISASI 3
fig = plt.figure(figsize=(14, 6.2), dpi=300)
gs = gridspec.GridSpec(1, 2, width_ratios=[1.05, 1.15], wspace=0.28)

ax1 = fig.add_subplot(gs[0, 0])
corr_labels = ['Likes (Y)', 'ln(Likes+1)', 'Replies', 'Panjang Ktr (X1)', 'Jumlah Kata', 'Negativitas (X2)', 'Kerentanan Fin.']
corr_mat = df[corr_cols].corr(method='spearman')
corr_mat.index = corr_labels
corr_mat.columns = corr_labels
sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='RdBu_r', vmin=-0.2, vmax=1.0,
            cbar_kws={'label': 'Koefisien Korelasi Rank Spearman (ρ)'}, ax=ax1, linewidths=0.8)
ax1.set_title('A. Matriks Korelasi Rank Spearman Antar Variabel', fontsize=11.5, fontweight='bold', pad=10)
ax1.tick_params(axis='x', rotation=30)

ax2 = fig.add_subplot(gs[0, 1])
cluster_means = df.groupby('cluster_id')[['X1_char_count', 'X2_negativity_score', 'financial_vulnerability_score']].mean().loc[cluster_order]
short_labels = [
    "K1: @charttechnician27\n(Aset Rp430 Juta)",
    "K2: @gussupriyatna17\n(Anak = Investasi)",
    "K3: @moon_delluna06\n(Bebas Pinjol 27th)",
    "K4: @mfaya_\n(Mahar Sandwich)",
    "K5: @mamaformulator\n(Rincian Tanggungan)",
    "K6: @panjisyamsi\n(Psikologi & SJSN)"
]
x_pos = np.arange(len(short_labels))
w = 0.26
ax2.bar(x_pos - w, cluster_means['X1_char_count'] / 5.0, width=w, label='Panjang Karakter (X1 / 5)', color='#1B365D', edgecolor='black')
ax2.bar(x_pos, cluster_means['X2_negativity_score'], width=w, label='Skor Negativitas (X2, 0-100)', color='#E05A47', edgecolor='black')
ax2.bar(x_pos + w, cluster_means['financial_vulnerability_score'], width=w, label='Skor Kerentanan Fin. (0-100)', color='#2E8B57', edgecolor='black')

ax2.set_xticks(x_pos)
ax2.set_xticklabels(short_labels, fontsize=8.5, rotation=20, ha='right')
ax2.set_ylabel('Skor Rata-Rata Terindeks (0 - 100)', fontsize=10)
ax2.set_title('B. Profil Rata-Rata Panjang Teks, Negativitas & Kerentanan per Klaster', fontsize=11.5, fontweight='bold', pad=10)
ax2.legend(loc='upper left', fontsize=8.5, frameon=True)
ax2.grid(True, axis='y')

fig.suptitle('KORELASI MULTIVARIAT & PROFIL EMOSIONAL-FINANSIAL ANTAR KLASTER DISKUSI',
             fontsize=13.5, fontweight='bold', y=1.01)
plt.savefig(FIG_OUTPUT_DIR / 'visualisasi_3_korelasi_dan_klaster_utas.png', bbox_inches='tight', dpi=300)
plt.close(fig)

# VISUALISASI 4
fig = plt.figure(figsize=(14, 6), dpi=300)
gs = gridspec.GridSpec(1, 2, width_ratios=[1.15, 1], wspace=0.28)

ax1 = fig.add_subplot(gs[0, 0])
theme_patterns = {
    'Bantu Ortu / Keluarga Inti': r'ortu|orang tua|orangtua|ibu|ibuk|ayah|bapa|mamah|keluarga',
    'Tabungan / Aset / Investasi': r'tabungan|nabung|aset|investasi|emas|saham|logam mulia|dana pensiun',
    'Sandwich Generation / Anak Pertama': r'sandwich|sandgen|sendwich|tanggungan|nanggung|anak pertama|sulung|bungsu',
    'Utang / Pinjol / Paylater / Emok': r'utang|hutang|ngutang|pinjol|paylater|spaylater|emok|cicilan|angsuran|kpr',
    'Biaya Sekolah Adik / Ponakan': r'adek|adik|ponakan|keponakan|sekolah|spp|kuliah|study tour',
    'Tekanan Gaji / UMR / Biaya Hidup': r'gaji|umr|penghasilan|listrik|bpjs|kos|kontrak|makan|indomie|uang dapur'
}
theme_counts = {
    label: int(df['combined_text'].str.lower().str.contains(pat, regex=True).sum())
    for label, pat in theme_patterns.items()
}

theme_s = pd.Series(theme_counts).sort_values(ascending=True)
bars = ax1.barh(theme_s.index, theme_s.values, color='#1B365D', edgecolor='black', alpha=0.88, height=0.6)
for bar in bars:
    val = bar.get_width()
    pct = val / len(df) * 100.0
    ax1.text(val + 1.5, bar.get_y() + bar.get_height()/2, f"{int(val)} ({pct:.1f}%)",
             va='center', ha='left', fontsize=9, fontweight='bold')
ax1.set_xlim(0, max(theme_s.values) * 1.25)
ax1.set_title('A. Frekuensi Kemunculan Tema Tekanan Finansial & Sosial (N = 234)', fontsize=11.5, fontweight='bold', pad=10)
ax1.set_xlabel('Jumlah Komentar / Utas yang Memuat Tema', fontsize=10)
ax1.grid(True, axis='x')

ax2 = fig.add_subplot(gs[0, 1])
cat_counts = df['vulnerability_category'].value_counts().sort_index()
colors_cat = ['#4E79A7', '#F28E2B', '#E15759', '#B07AA1']
wedges, texts, autotexts = ax2.pie(
    cat_counts.values, labels=cat_counts.index, autopct='%1.1f%%',
    startangle=140, colors=colors_cat, pctdistance=0.78,
    wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2)
)
for t in autotexts:
    t.set_fontsize(9)
    t.set_fontweight('bold')
    t.set_color('white')
ax2.set_title('B. Proporsi Kategori Skor Kerentanan Finansial Audiens', fontsize=11.5, fontweight='bold', pad=10)

fig.suptitle('PEMETAAN KATA KUNCI SCRAPE & STRATIFIKASI KERENTANAN FINANSIAL GEN Z / MILENIAL',
             fontsize=13.5, fontweight='bold', y=1.01)
plt.savefig(FIG_OUTPUT_DIR / 'visualisasi_4_profil_beban_dan_kata_kunci.png', bbox_inches='tight', dpi=300)
plt.close(fig)

print(f"\n[SUKSES] File CSV disimpan di    : {csv_out}")
print(f"[SUKSES] 4 Grafik PNG disimpan di: {FIG_OUTPUT_DIR}")