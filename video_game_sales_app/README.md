# Video Game Sales Prediction App

Aplikasi ini memuat dataset Global Video Game Sales dari sumber eksternal dan melatih model regresi linear untuk memprediksi `Other_Sales` berdasarkan Tahun rilis, Genre, dan penjualan Amerika Utara, Eropa, serta Jepang. Total `Global_Sales` dihitung sebagai jumlah input pasar besar plus prediksi `Other_Sales`.

## Struktur Folder

```
video_game_sales_app/
├── video_game_sales_service.py
├── video_game_sales_web.py
├── requirements.txt
├── README.md
└── templates/
    └── index_vg.html
```

## Instalasi

1. Buka terminal di folder `video_game_sales_app`
2. Pasang dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Menjalankan Aplikasi

Sebelum menjalankan, pastikan dataset tersedia di lokasi eksternal misalnya:
`D:\vgsales.csv\vgsales.csv`

Jalankan service model:
```bash
python video_game_sales_service.py
```

Jalankan web app:
```bash
python video_game_sales_web.py
```

Buka browser ke: `http://127.0.0.1:5002`

## Catatan

- Dataset video game berada di luar folder aplikasi.
- Jika dataset berada di lokasi lain, set environment variable `VGSales_CSV_PATH` sebelum menjalankan.
