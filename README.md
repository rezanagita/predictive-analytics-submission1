# Laporan Proyek Machine Learning - REZA NAGITA NURHAZIZAH

## Domain Proyek

### Latar Belakang
Pemilihan perusahaan asuransi merupakan keputusan penting, terutama di tengah maraknya kasus gagal bayar yang terjadi dalam industri asuransi di Indonesia. Gagal bayar, yakni ketidakmampuan perusahaan asuransi untuk memenuhi kewajibannya dalam membayarkan klaim kepada nasabah, dipicu oleh berbagai faktor. Di antaranya adalah lemahnya manajemen internal, kurangnya kehati-hatian dalam pengelolaan dana, ketidaktransparanan informasi produk, tekanan likuiditas, hingga lemahnya pengawasan oleh lembaga terkait (Malasari et al., 2020). Kondisi ini tidak hanya merugikan nasabah, tetapi juga berdampak pada menurunnya kepercayaan publik serta berpotensi mengganggu stabilitas ekonomi nasional [1].

Dalam konteks tersebut, kemampuan perusahaan asuransi dalam melakukan penilaian risiko dan prediksi premi secara akurat menjadi sangat krusial untuk menghindari kerugian finansial jangka panjang. Terlebih lagi, dengan meningkatnya biaya layanan medis dan kesadaran masyarakat terhadap pentingnya perlindungan kesehatan, sistem asuransi dituntut untuk menetapkan premi secara cermat dan adil. Premi yang terlalu tinggi dapat membuat calon nasabah enggan membeli polis, sementara premi yang terlalu rendah dapat menyebabkan defisit keuangan ketika klaim meningkat. Oleh karena itu, dibutuhkan mekanisme prediktif yang andal untuk memperkirakan biaya klaim dan menetapkan premi secara proporsional.

Salah satu pendekatan yang efektif untuk menjawab tantangan tersebut adalah penerapan metode machine learning, khususnya algoritma regresi. Metode ini mampu menangani kompleksitas dan keragaman data nasabah, serta memodelkan hubungan non-linear antara variabel risiko seperti usia, indeks massa tubuh (BMI), status merokok, dan jumlah tanggungan, dengan biaya premi yang harus dibayarkan. Penelitian oleh Mosleh (2023) menunjukkan bahwa pendekatan berbasis machine learning tidak hanya meningkatkan akurasi prediksi premi, tetapi juga memungkinkan personalisasi produk asuransi sesuai dengan profil risiko masing-masing individu [2].

Referensi:

[1] A. Malasari, M. Adam, dan H. Hanafi, “Faktor-Faktor Penyebab Terjadinya Gagal Bayar pada Perusahaan Asuransi di Indonesia,” Jurnal Hukum dan Pembangunan Ekonomi, vol. 8, no. 1, pp. 11–20, 2020. [Online]. Tersedia: https://ejournal2.undip.ac.id/index.php/jphi/article/view/20657/10810

[2] M. Mosleh, Predicting health insurance premiums using machine learning: A novel regression-based model for enhanced accuracy and personalization, Academia.edu, 2023. Tersedia: https://www.academia.edu/127732009/

---

## Business Understanding
### Problem Statements

1. Bagaimana memodelkan hubungan antara karakteristik individu (usia, BMI, status merokok, dll.) dengan biaya premi asuransi kesehatan?
Permasalahan ini muncul karena faktor-faktor penentu premi bersifat kompleks dan tidak selalu linear, sehingga sulit diakomodasi oleh pendekatan konvensional.

2. Bagaimana meningkatkan akurasi prediksi biaya premi untuk meminimalkan risiko gagal bayar perusahaan asuransi?
Ketidakakuratan dalam prediksi premi dapat mengakibatkan kerugian dan menurunnya kepercayaan masyarakat terhadap lembaga asuransi.

3. Bagaimana memanfaatkan data historis premi asuransi untuk menghasilkan sistem rekomendasi premi yang dapat disesuaikan dengan profil risiko pengguna secara otomatis?
Pendekatan personalisasi diperlukan agar perusahaan dapat menyasar segmen pasar secara lebih efisien dan adil.

### Goals

1. Mengembangkan model regresi prediktif untuk mengidentifikasi pengaruh variabel-variabel individu terhadap biaya premi asuransi kesehatan.
Model ini akan membantu dalam memahami seberapa besar pengaruh setiap faktor terhadap biaya asuransi.

2. Meningkatkan akurasi prediksi premi asuransi menggunakan algoritma machine learning dibandingkan dengan metode konvensional.
Dengan akurasi yang tinggi, perusahaan asuransi dapat menetapkan premi yang mencerminkan risiko sesungguhnya dan menghindari kerugian finansial.

3. Menyediakan sistem prediksi premi yang fleksibel dan dapat dipersonalisasi sesuai karakteristik masing-masing pengguna.
Hal ini mendukung terciptanya layanan asuransi yang lebih inklusif, transparan, dan adil.

### Solution statements
Upaya dalam mencapai tujuan proyek, dengan mengimplementasikan solusi sebagai berikut:
1. Explanatory Data Analysis
   - Mengidentifikasi sebaran data (distribusi umur, BMI, jumlah anak, status merokok, dll.)
   - Melihat korelasi antar variabel untuk mengetahui fitur mana yang paling berpengaruh terhadap biaya asuransi.
2. Regresi dengan algoritma machine learning
   Pada tahap ini, menggunakan beberapa algoritma machine learning untuk melihat mana yang memberikan performa terbaik dalam konteks prediksi biaya premi:
   - Linear Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
4. Melakukan hyperparameter tuning untuk meningkatkan kinerja model baseline,
   dilakukan dengan menggunakan Grid Search untuk menemukan kombinasi hyperparameter optimal (misal: jumlah neuron, kedalaman pohon, learning rate, dll.)
5. Evaluasi dan pengujian model prediksi
   Setelah model dilatih dan dioptimalkan, kinerjanya akan dievaluasi menggunakan metrik regresi:
   - Mean Absolute Error (MAE) – mengukur rata-rata kesalahan absolut prediksi.
   - Mean Squared Error (MSE) – lebih sensitif terhadap kesalahan besar.
   - R² Score – mengukur proporsi variansi target yang dijelaskan oleh model.

---

## Data Understanding
### Karakteristik Dataset
Dataset yang digunakan pada proyek ini diambil dari kaggle: 
[Health Insurance](https://www.kaggle.com/datasets/denywisnu/health-insurance-dataset/data).
Dataset ini terdiri dari 1338 baris data dan 7 fitur, yang mencakup informasi demografis, gaya hidup, serta nilai biaya asuransi yang harus dibayarkan. Berikut penjelasan setiap variabel:

| Variabel | Keterangan |
| ------ | ------ |
| Age | Usia pemegang polis |
| Sex | Jenis kelamin dari pemegang polis, dengan dua kemungkinan: male dan female |
| BMI | Body Mass Index (berat badan dalam kg dibagi tinggi badan kuadrat dalam meter). Merupakan indikator obesitas dan salah satu faktor penting dalam penilaian risiko kesehatan. |
| Children | Jumlah anak yang menjadi tanggungan dalam asuransi.  |
| Smoker | Status merokok dari individu, terdiri dari yes dan no. Ini adalah fitur penting yang sangat memengaruhi premi asuransi. |
| Region | Lokasi tempat tinggal pemegang polis, terdiri dari empat wilayah: southeast, southwest, northeast, dan northwest. |
| Charges | Total biaya asuransi yang harus dibayarkan oleh pemegang polis. Ini adalah variabel target (label) dalam proyek ini. |

### Statistik Deskriptif
- Dataset ini tidak memiliki missing value,
- Memiliki 1 duplikasi data, sehingga perlu dilakukan hapus duplikat,
- Data kategori (sex, smoker, region) akan dilakukan One Hot Encoding sebelum ketahapan modeling,
- Perlu penanganan outlier pada data numerik (BMI, Charges)
  
- **Age**:
  - Rata-rata: 39 tahun
  - Minimum: 18 tahun
  - Maksimum: 64 tahun
- **BMI**:
  - Rata-rata: 30 kg
  - Minimum: 15 kg
  - Maksimum: 53 kg
- **Children**:
  - Rata-rata: 1 anak
  - Minimum: 0 anak
  - Maksimum: 5 anak
- **Charges**:
  - Rata-rata: 13.279 USD
  - Minimum: 1.121 USD
  - Maksimum: 63.770 USD

### Distribusi Data
1. **Charges**:
   - Distribusi charges menunjukkan skewed ke kanan (positively skewed), mengartikan charges rendah < 20.000 memiliki frekuensi jauh lebih tinggi mencapai > 200 dibanding charges dengan nilai > 20.000
   - Mayoritas orang membayar biaya asuransi dalam kisaran rendah hingga menengah (sekitar 0–20.000), tapi ada beberapa yang membayar sangat tinggi (> 40.000).
  
3. **age**: 
   - semakin tua usia seseorang, semakin tinggi charges yang dibayarkan.
   - Ada orang muda dengan biaya tinggi, tapi sedikit jumlahnya (outlier). 

4. **BMI**:
   - Terdapat sebaran biaya yang tinggi untuk individu dengan BMI tinggi, tapi itu juga tidak konsisten.
     
5. **Children**:
   - Visualisasi distribusi children menunjukkan tidak terlalu memengaruhi charges secara langsung.
   - Distribusi charges relatif menyebar pada setiap nilai children (0 hingga 5).
   - Ini menyiratkan bahwa jumlah tanggungan anak mungkin bukan faktor utama dalam menentukan biaya asuransi, atau efeknya kecil.
     
7. **sex**:
   - Distribusi jumlah male dan female tampak seimbang sekitar 650 untuk female dan > 650 untuk male 

8. **smoker**:
   - Total kategori yang tidak merokok ( >1000 ) berbenading jauh dengan yang merokok(hanya sekitar  200)
     
9. **region** empat wilayah utama dengan distribusi sebagai berikut:
   - Southeast: 360 nasabah
   - Southwest: 340 nasabah
   - Nortwest : 340 nasabah
   - Northeast: 340 nasabah
---

## Data Preparation
### Langkah-langkah melakukan data preparation
1. **Drop duplikasi data** karena didapatkan 1 duplikasi data maka perlu di hapus
2. **One Hot Encoding** pada data kategorikal sex, smoker, dan region
3. **Menangani outlier** menggunakan IQR ((Interquartile Range)
4. **Features Engineering** untuk melihat relasi kuat antara fitur dengan membuat fitur baru:
  -  age × smoker → Efek usia mungkin jauh lebih besar jika seseorang merokok.
  -  bmi × smoker → Perokok dengan BMI tinggi bisa punya risiko lebih tinggi.
  -  age × bmi → Semakin tua dan semakin tinggi BMI, mungkin berisiko lebih mahal.
6. **Standarisasi data numerik** diterapkan pada fitur numerik age,bmi,children,charges menggunakan standardscaler

### Alasan melakukan data preparation
   *  Drop duplikasi penting dilakukan pada tahapan awal dalam preparation data karena menyebabkan bias dalam model. ini akan mempengaruhi akurasi prediksi dan distribusi statistik.
   *  One hot encoding dilakukan untuk mengubah data aktegorikal ke numerikal karena machine learning tidak bisa membaca/ mengolah data numerikal
   *  Outlier perlu ditangani karena  bisa mengganggu model terutama model berbasis jarak atau regresi yang akan dilakukan dalam proyek ini, tujuannya untuk meningkatkan akurasi model dan stabilitas training.
   *  feature engineering membantu model menangkap hubungan non-linear yang mungkin tidak terlihat dari fitur asli.
   *  standarisasi membantu model berkonvergensi lebih cepat saat training dan menghindari dominasi satu fitur terhadap yang lain.
- Proses data preparation ini sangat penting dan berguna sebelum masuk ketahapan modeling data supaya membuat hasil analisis data menjadi lebih akurat dan tidak bias.

## Modeling
### Algoritma Regresi yang digunakan dalam proyek ini
1. Split dataset menjadi data train dan test dengan perbandingan 80:20
   Jumlah data:  1190
   Jumlah data latih:  952
   Jumlah data test:  238
2. Melatih model dengan 3 algoritma
   
      a. **Linear Regresion** (sebagai baseline)
         - memiliki performa yang cukup baik (R²: 0.559498), tetapi kalah dari GBR. Ini menunjukkan bahwa hubungan antara fitur (usia, BMI, status merokok, jumlah tanggungan) dan biaya premi kemungkinan tidak sepenuhnya linier.
   
      b. **Random Forest Regressor**
   
      | Parameter           | Nilai  | Deskripsi                                                     |
      | ------------------- | ------ | ------------------------------------------------------------- |
      | `random_state`      | 42     | Penetapan seed agar hasil dapat direproduksi.                 |
      | `n_estimators`      | 200    | Jumlah pohon (trees) dalam hutan.                             |
      | `max_depth`         | 10     | Kedalaman maksimum tiap pohon. Membantu mencegah overfitting. |
      | `min_samples_split` | 10     | Minimum jumlah sampel untuk memisahkan node.                  |
      | `min_samples_leaf`  | 2      | Minimum jumlah sampel di node daun.                           |
      | `max_features`      | 'sqrt' | Jumlah maksimum fitur yang dipertimbangkan saat membagi node. |
      | `bootstrap`         | True   | Menggunakan bootstrap sampling untuk membangun tiap pohon.    |

      c. **Gradient Boosting Regressor**
   
| Parameter           | Nilai  | Deskripsi                                                             |
| ------------------- | ------ | ----------------------------------------------------------------------|
| `n_estimators`      | 300    | Jumlah pohon dalam boosting stages.                                           |
| `learning_rate`     | 0.03   | Kontribusi tiap pohon terhadap prediksi akhir. Semakin kecil, semakin stabil. |
| `max_depth`         | 4      | Kedalaman maksimum pohon individual.                                          |
| `min_samples_split` | 10     | Minimum jumlah sampel untuk memisahkan node.                                  |
| `min_samples_leaf`  | 3      | Minimum jumlah sampel di node daun.                                           |
| `subsample`         | 0.8    | Persentase data yang digunakan untuk tiap pohon. Cegah overfitting.           |
| `max_features`      | 'sqrt' | Jumlah maksimum fitur yang digunakan untuk split.                             |
| `random_state`      | 42     | Seed agar hasil bisa direproduksi.                                            |

### Hasil evaluasi model 
![evaluasi dari 3 model](https://github.com/user-attachments/assets/fcba755a-7a41-470f-a186-e9a2fb3677d0)

Model Random Forest Regression dan Gradient Boosting Regression memiliki selisih yang sangat kecil,pada proyek ini saya menggunakan model Gradient Boosting Regression alasannya:

- MAE (terkecil): 0.339 → paling sedikit kesalahan prediksi rata-rata.

- MSE dan R² sangat kompetitif dengan Random Forest (hanya selisih kecil).
3. Hyperparameter Tuning dengan Grid Search pada model Gradient Boosting Regression memperoleh 
**Best Parameters GBR: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'subsample': 1.0}
Best R² GBR: 0.6528779522578873** dan hasil evaluasi akhir 
**R² Test: 0.5922835600328693
MAE Test: 0.34408043755403495
MSE Test: 0.407566556286526**

ini menunjukkan peningkatan pada R² menjadi lebih tinggi (0.592) dibanding dengan model latih GBR pertama R² (0.572)
  
## Evaluation
| Metrik                                     | Penjelasan                                                                                                                  |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| **MAE (Mean Absolute Error)**              | Mengukur rata-rata kesalahan absolut antara nilai prediksi dan nilai aktual. Semakin kecil MAE, semakin baik akurasi model. |
| **MSE (Mean Squared Error)**               | Sama seperti MAE, tetapi memberikan penalti lebih besar terhadap kesalahan yang besar. Berguna untuk menangkap outlier.     |
| **R² (R-Squared / Koefisien Determinasi)** | Menunjukkan proporsi variasi target yang dapat dijelaskan oleh model. Nilai R² mendekati 1 menunjukkan model yang baik.     |

| Model                                            | MAE    | MSE    | R² Score  |
| ------------------------------------------------ | ------ | ------ | --------- |
| **Linear Regression**                            | 0.3652 | 0.4403 | 0.5595    |
| **Random Forest Regressor**                      | 0.3466 | 0.4248 | 0.5751    |
| **Gradient Boosting Regressor** (sebelum tuning) | 0.3394 | 0.4269 | 0.5729    |
| **Gradient Boosting Regressor** (setelah tuning) |*0.3440*|*0.4075*| **0.5922**|

## Conclution
- Penentuan pembayaran charges ini dapat tinggi seiring dengan usia dan dan status perokok(aktif).
- Penentuan charges juga dapat dipengaruhi oleh BMI dan status perokok, semakin tinggi BMI dan perokok aktif maka biaya charges bisa lebih tinggi
- Penentuan charges berdasarkan tanggungan anak, semakin banyak anak biaya yang dibayarkan lebih rendah dibanding dengan tidak memiliki anak charges lebih tinggi terlebih perokok aktif
### Jawaban dari Problem statement
**1. Bagaimana memodelkan hubungan antara karakteristik individu (usia, BMI, status merokok, dll.) dengan biaya premi asuransi kesehatan?**
pemodelan dilakukan dengan membandingan tiga algoritma regresi dan model paling baik adalah Gradient Boosting Regressor karena dapat menilai ketidaklinearitasan fitur terhadap target(charges) dan memiliki nilai kesalahan prediksi terkecil dibanding dengan model lainnya.

**2. Bagaimana meningkatkan akurasi prediksi biaya premi untuk meminimalkan risiko gagal bayar perusahaan asuransi?**
Pelatihan pertama dengan ketiga model masih belum menghasilkan tingkat evaluasi yang baik, untuk meningkatkan nilai akurasi prediksi dengan melakukan hyperparameter tuning terhadap algoritma Gradient Boosting Regressor dan berhasil meningkat hingga 0.5922

**3. Bagaimana memanfaatkan data historis premi asuransi untuk menghasilkan sistem rekomendasi premi yang dapat disesuaikan dengan profil risiko pengguna secara otomatis?**
Model yang telah dilatih pada data historis mampu mempelajari pola hubungan antara karakteristik pengguna dan biaya premi. Dengan memasukkan data individu seperti usia, BMI, jumlah anak, status merokok, dan wilayah, model dapat memprediksi besarnya premi secara otomatis dan personal.

## Saran pengembangan
Melakukan analisis penilaian resiko lebih lanjut dengan menambahkan fitur yang relavan seperti riwayat kesehatan ataupun gaya hidup, sehingga dapat memberikan model belajar dari data yang lebih banyak dan dapat meningkatkan akurasi jauh lebih baik 
