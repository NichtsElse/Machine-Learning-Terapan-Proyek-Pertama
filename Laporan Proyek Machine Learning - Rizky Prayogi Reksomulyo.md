# Laporan Proyek Machine Learning - Rizky Prayogi Reksomulyo

## Domain Proyek Kesehatan

Kanker payudara merupakan salah satu masalah kesehatan global yang paling mendesak. WHO mencatat pada tahun 2022 ada 2,3 juta wanita di seluruh dunia yang didiagnosis menderita kanker payudara, dengan angka kematian mencapai 670.000 kasus[1]. Berdasarkan data dari Globocan 2022 Kanker payudara menempati urutan pertama terkait jumlah kanker terbanyak di Indonesia jumlah kasus baru kanker payudara mencapai 66.271 kasus (16,2%) dari total 408.661 kasus baru kanker di Indonesia. Sementara itu, untuk jumlah kematiannya mencapai lebih dari 22 ribu jiwa[2]. Kanker ini dapat menyerang wanita di berbagai usia setelah pubertas, namun prevalensinya meningkat seiring bertambahnya usia. Fakta ini menunjukkan bahwa kanker payudara adalah ancaman serius yang membutuhkan perhatian khusus, terutama dalam hal deteksi dini dan penanganan yang efektif.

berdasarkan data diatas saya memiliki tujuan untuk mengembangkan model prediktif yang dapat membantu mendeteksi apakah kanker bersifat jinak (benign) atau ganas (malignant). Deteksi dini sangat penting dalam penanganan kanker payudara, karena semakin cepat kanker terdeteksi, semakin tinggi kemungkinan kesembuhan dan penurunan risiko kematian. Dengan menggunakan dataset diagnostik Kanker Payudara Wisconsin dan algoritma machine learning, proyek ini diharapkan dapat memberikan solusi yang lebih cepat, akurat, dan efisien dalam membantu diagnosis kanker payudara. 

Hasil proyek ini dapat berkontribusi pada peningkatan akses dan kualitas diagnosis di berbagai negara, terutama di wilayah dengan sumber daya terbatas, sehingga membantu mengurangi angka kematian pada penderita kanker payudara yang selaras dengan tujuan WHO Global Breast Cancer Initiative (GBCI) yaitu mengurangi angka kematian akibat kanker payudara global sebesar 2,5% per tahun, sehingga mencegah 2,5 juta kematian akibat kanker payudara di seluruh dunia antara tahun 2020 dan 2040.
 
## Business Understanding
### Problem Statements
- Bagaimana cara memprediksi apakah seseorang memiliki kanker payudara jinak (benign) atau ganas(malignant)?

### Goals
- Memprediksi apakah seseorang memiliki kanker payudara jinak(benign) atau ganas(malignant) berdasarkan statistik kesehatannya.

### Solution statements
- Menggunakan metode Random Forest, di mana metode ini merupakan algoritma ensemble yang powerful dan mampu menangani data yang kompleks serta fitur yang saling berinteraksi. Random Forest juga memberikan hasil yang stabil dan dapat diandalkan, sehingga digunakan sebagai pembanding dari Logistic Regression.

- Menggunakan Decision Tree yang efektif untuk klasifikasi dan mudah dipahami, memprediksi kanker payudara berdasarkan fitur yang paling informatif. Lalu, AdaBoost digunakan untuk meningkatkan performa dengan menggabungkan beberapa weak learners, memperbaiki kesalahan prediksi, dan meningkatkan akurasi, menjadikannya model yang cocok sebagai pembanding.

- Sebagai metrik pembanding, menggunakan beberapa metrik evaluasi antara lain Akurasi, Precision, Recall, dan F1-Score untuk mendapatkan gambaran yang lebih komprehensif mengenai performa model dalam mendeteksi kanker jinak(benign) dan ganas(malignant).

## Data Understanding
data yang digunakan adalah Breast Cancer Wisconsin (Diagnostic) yang bersumber di kaggle. Dataset ini terdiri dari 569 baris data, dan memiliki 32 kolom data dan dapat diperoleh dari [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).

### Variabel-variabel pada dataset adalah sebagai berikut:
1.  ID number
2.  Diagnosis (M = Malignant(ganas), B = Benign(jinak))
3.  10 fitur utama untuk setiap inti sel:
 a. radius (rata-rata jarak dari pusat ke titik-titik di keliling)
 b. texture (standar deviasi nilai gray-scale)
 c. perimeter(keliling)
 d. area
 e. smoothness (variasi lokal dalam panjang radius)
 f. compactness (perimeter^2 / area - 1.0)
 g. concavity (tingkat keparahan bagian cekung pada kontur)
 h. concave points (jumlah bagian cekung dari kontur)
 i. symmetry
 j. fractal dimension ("perkiraan tepi" - 1)
mean, standard error dan "worst" dari fitur-fitur ini dihitung untuk 10 fitur utama menghasilkan 30 fitur.

### Exploratory Data Analysis
pada proyek ini terdapat beberapa visualisasi seperti pada dibawah yaitu bar chart data diagnosis
![presisi!](https://github.com/NichtsElse/Machine-Learning-Terapan-Proyek-Pertama/blob/main/bar1.png)

visualisasi korelasi antar variabel pada heatmap
![presisi!](https://github.com/NichtsElse/Machine-Learning-Terapan-Proyek-Pertama/blob/main/cor.png)

visualisasi pairplot pada bagian variabel mean
![presisi!](https://github.com/NichtsElse/Machine-Learning-Terapan-Proyek-Pertama/blob/main/mean.png)

visualisasi pairplot pada bagian variabel standard error
![presisi!](https://github.com/NichtsElse/Machine-Learning-Terapan-Proyek-Pertama/blob/main/se.png)

visualisasi pairplot pada bagian variabel worst
![presisi!](https://github.com/NichtsElse/Machine-Learning-Terapan-Proyek-Pertama/blob/main/worst.png)

visualisasi outliers pada bagian variabel mean
![presisi!](https://github.com/NichtsElse/Machine-Learning-Terapan-Proyek-Pertama/blob/main/outliermean.png)

visualisasi pairplot pada bagian variabel standard error
![presisi!](https://github.com/NichtsElse/Machine-Learning-Terapan-Proyek-Pertama/blob/main/outlierse.png)

visualisasi pairplot pada bagian variabel worst
![presisi!](https://github.com/NichtsElse/Machine-Learning-Terapan-Proyek-Pertama/blob/main/outlierworst.png)

## Data Preparation
Dalam data preparation, dilakukan beberapa hal  sebelum memasukkan data ke model latih yaitu:

- Label Encoder
teknik ini digunakan untuk mengubah data kategorikal (label) menjadi data numerik.

- Handling Outlier
Handling outlier berfungsi untuk meningkatkan akurasi, mencegah overfitting, dan membuat model lebih stabil serta mudah diinterpretasikan. Dengan menangani data ekstrem, model fokus pada pola yang lebih representatif, sehingga hasil prediksi lebih konsisten. 
- Train-Test-Split
proses ini berguna untuk membagi dataset menjadi data training dan testing pembagian data pada proyek ini ada 80:30.

- Standarisasi
Proses  ini dilakukan untuk Meningkatkan performa algoritma machine learning. Standarisasi ini diterapkan pada kolom-kolom yang memiliki fitur numerik.

## Modeling
Model yang saya gunakan pada proyek ini yaitu:
- Adaptive Boosting
Menggunakan Decision Tree yang merupakan algoritma yang membangun model klasifikasi dengan memecah data berdasarkan fitur yang paling relevan untuk membuat keputusan. Pada proyek ini, model Decision Tree digunakan dengan kedalaman maksimum(max_depth) sebesar 3, yang berarti pohon keputusan dibatasi hingga tiga tingkat untuk mencegah overfitting. Lalu menggunakan AdaBoost yang merupakan metode boosting yang meningkatkan performa model dengan menggabungkan beberapa weak learners (seperti Decision Tree) menjadi model yang lebih kuat. Pada proyek ini, AdaBoost digunakan dengan estimator berupa model Decision Tree yang memiliki kedalaman maksimum 3 dan jumlah estimators sebanyak 1000. Hal ini membantu model belajar dari kesalahan prediksi sebelumnya, meningkatkan akurasi secara keseluruhan.

- Random Forest
Random Forest adalah algoritma ensemble yang mengombinasikan hasil dari beberapa pohon keputusan (Decision Tree) untuk menghasilkan satu prediksi akhir. Algoritma ini menggunakan proses bagging atau bootstrap aggregating, di mana setiap pohon dilatih menggunakan subset data yang berbeda. Pada proyek ini, model Random Forest menggunakan 100 n_estimators dan dibatasi dengan kedalaman maksimum 
3 untuk menjaga keseimbangan antara bias dan variansi.

Saya mengambil data weighted average dikarenakan data imbalance yang mana data kanker jinak (benign) lebih banyak daripada kanker ganas(malignant). Berdasarkan hasil train AdaBoost lebih unggul dalam hal menangani kesalahan prediksi dan akurasi pada dataset daripada Random Forest dengan metrik presisi 1%, akurasi, recall dan F1 score lebih tinggi 2%.

## Evaluation
Metrik evaluasi yang digunakan  adalah Confusion Matrix yang merupakan sebuah teknik yang digunakan dalam data mining dan machine learning untuk menghitung seberapa baik sebuah model dapat memprediksi label dari sebuah data seperti contoh pada gambar dibawah.
![presisi!](https://miro.medium.com/v2/resize:fit:750/format:webp/1*f5ZeXvhsNFZ4q91M4Lotgg.jpeg)
Selanjutnya saya akan membahas secara rinci mengenai metrik akurasi, precision, recall, dan F1-score sebagai berikut:
1. Akurasi
Akurasi adalah persentase prediksi yang benar dari keseluruhan prediksi yang dilakukan oleh model. akurasi dapat dihitung menggunakan rumus dibawah.
![presisi!](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*gFW6rXbctrhWHxD8OXi4wg.png)
 
2. Precision (Presisi)
Precision adalah persentase prediksi yang benar dari semua prediksi yang positif. Dengan kata lain, precision mengukur akurasi dari prediksi positif model. Precision dapat dihitung menggunakan rumus dibawah.
![presisi!](https://miro.medium.com/v2/resize:fit:828/format:webp/1*VXnUvOEdf3IiYVCD6Wd2vg.png)

3. Recall (Sensitivitas)
Recall adalah persentase dari semua kasus positif yang terdeteksi oleh model. Ini mengukur seberapa baik model dapat menangkap semua kasus yang benar-benar positif. Recall dapat dihitung menggunakan rumus dibawah.
![presisi!](https://miro.medium.com/v2/resize:fit:828/1*OV0hfgCStTI8hy6lAY1SdA.jpeg)

4. F1-Score
F1-score adalah rata-rata harmonis dari precision dan recall. Metrik ini berguna ketika Anda ingin mencapai keseimbangan antara precision dan recall. F1-Score dapat dihitung menggunakan rumus dibawah.
![presisi!](https://miro.medium.com/v2/resize:fit:1100/1*tEck5hzpmrv7lfnjiT7DhQ.jpeg)

Berdasarkan hasil evaluasi menggunakan teknik yang dijelasakan sebelumnya kedua model mendapatkan hasil metrik sebagai berikut:
1. Random Forest
Accuracy    : 96%
Precision   : 97% 
Recall      : 96%
F1 Score    : 96%

2. Adaptive Boasting(Decision Tree + Adaboost)
Accuracy    : 97%
Precision   : 97% 
Recall      : 97%
F1 Score    : 97%

berdasarkan hasil evaluasi diatas model Adaptive Boosting adalah model terbaik untuk mengklasifikasi kanker payudara

## Daftar Pustaka
[1] Ferlay J, Ervik M, Lam F, Laversanne M, Colombet M, Mery L, Piñeros M, Znaor A, Soerjomataram I, Bray F (2024). Global Cancer Observatory: Cancer Today. Lyon, France: International Agency for Research on Cancer. Available from: https://gco.iarc.who.int/today, accessed 22 October 2024.
[2] World Health Organization (2024). Breast Cancer. Online at https://www.who.int/news-room/fact-sheets/detail/breast-cancer, accessed 22 October 2024
