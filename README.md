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
   - Multi-Layer Perceptron (MLP) Regressor
4. Melakukan hyperparameter tuning dan cross-validation untuk meningkatkan kinerja model baseline,
   dilakukan dengan menggunakan Grid Search atau Randomized Search untuk menemukan kombinasi hyperparameter optimal (misal: jumlah neuron, kedalaman pohon, learning rate, dll.)
   dan K-Fold Cross Validation akan diterapkan agar performa model lebih stabil dan tidak bergantung pada satu set data latih dan uji saja.
6. Evaluasi dan pengujian model prediksi
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
   - Southeast: 350 nasabah
   - Southwest: 340 nasabah
   - Nortwest : 340 nasabah
   - Northeast: 340 nasabah

