# Analisis Statistika: Sandwich Generation & Tekanan Finansial Gen Z di Threads

Proyek analisis data kuantitatif dan visualisasi teks untuk membedah fenomena *sandwich generation* serta tekanan finansial Gen Z/Milenial di Indonesia berdasarkan $N = 234$ unggahan dan komentar Threads.

## Fokus Analisis
1. **Uji Deskriptif & Skewness**: Mengukur rata-rata, median, IQR, dan kemencengan distribusi (*skewness*) pada **Skor Kerentanan Finansial**, panjang karakter komentar, skor negativitas, dan jumlah *Likes*.
2. **Regresi Linier Berganda (OLS & Robust HC3)**: Memprediksi jumlah *Likes* ($Y$) dan transformasi logaritmik $\ln(\text{Likes} + 1)$ berdasarkan **Panjang Karakter Komentar ($X_1$)** dan **Skor Sentimen/Negativitas ($X_2$)**.

## Struktur Folder
```text
sandwich-gen-threads-analytics/
├── data/
│   ├── raw/
│   │   └── merged-1794142463790.json
│   └── processed/
│       └── data_olahan_threads_sandwich_gen.csv
├── src/
│   └── analisis_sandwich_generation.py
├── output/
│   ├── figures/
│   │   ├── visualisasi_1_distribusi_skewness_kerentanan.png
│   │   ├── visualisasi_2_regresi_linier_berganda.png
│   │   ├── visualisasi_3_korelasi_dan_klaster_utas.png
│   │   └── visualisasi_4_profil_beban_dan_kata_kunci.png
└── README.md
```

## Cara Menjalankan
1. Instal pustaka yang dibutuhkan:
   ```bash
   pip install numpy pandas scipy matplotlib seaborn
   ```
2. Jalankan skrip analisis dari folder utama proyek atau dari dalam folder `src/`:
   ```bash
   py src/analisis_sandwich_generation.py
   ```
3. Hasil olahan CSV akan tersimpan otomatis di `data/processed/` dan 4 grafik visualisasi beresolusi 300 DPI di `output/figures/`.

## Sumber Data
* Data komentar Threads diekstrak menggunakan perangkat lunak sumber terbuka [jobas5/thread-scrapper](https://github.com/jobas5/thread-scrapper).
