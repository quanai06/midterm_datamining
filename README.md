# Phân cụm khách hàng thương mại điện tử — Giữa kỳ Phân tích & Khai phá dữ liệu

Bài toán **Unsupervised (Clustering)**, không có nhãn mục tiêu. MSSV 24022433 → N = 260, 14 thuộc tính, 15 ngoại lai.
Pipeline: sinh dữ liệu → Data Understanding → Preprocessing → Preparation → Clustering (so sánh trước/sau xử lý).

## Cấu trúc
```
data/01_generated/    4 file đối chiếu + data_dictionary.csv (câu a)
  data_clean_reference.csv   D0 sạch (chỉ đối chiếu)
  data_after_noise.csv       D1 = D0 + nhiễu
  data_after_outliers.csv    D2 = D1 + ngoại lai
  data_raw_corrupted.csv     D3 = D2 + giá trị thiếu  <- đầu vào duy nhất của câu b-e
data/hidden/          ground_truth.csv (nhãn ẩn, chỉ dùng đánh giá ở câu e), injection_log.csv (nhật ký chèn nhiễu/ngoại lai/thiếu)
data/02_preprocessed  sau tiền xử lý (câu c): dữ liệu, nhật ký ô thay đổi, cờ, biến thể category, FROZEN_SHA256.json
data/03_prepared      sau chuẩn bị (câu d): X_main.csv (260x19), pipeline/, variants/ (full_features, pca90, category, Gower)
src/                  mã nguồn; design_validation/ là bằng chứng hiệu chỉnh thiết kế
notebooks/            01_generate_data_colab.ipynb chạy câu a trên Colab
figures/                    02_understanding (câu b), 03_preprocessing (Before/After), 03_preprocessing_eval (đối chiếu D0); reports/tables/ bảng số liệu
docs/                 data_dictionary.md, requirements_checklist.md, ai_log.md
data/04_clustering    nhãn cụm, tâm cụm, cấu hình, mã băm (câu e)
logs/                 log chạy từng script (có kết quả kiểm tra nghiệm thu)
```

## Chạy câu a
Local (CPU, vài giây, không cần GPU):
```
pip install -r requirements.txt
python src/01_generate_data.py          # sinh D0-D3, ghi data/ và logs/, chạy 42 kiểm tra nghiệm thu
```
Mã thoát 0 nghĩa là mọi kiểm tra đạt. Colab: mở `notebooks/01_generate_data_colab.ipynb`.

Câu b: `python src/02_data_understanding.py` (đọc D3 trước, đối chiếu D0-D2 và `data/hidden/` ở Phần B); kết luận trong `docs/cau_b_findings.md`.

Câu c: `python src/03_preprocess.py` (chỉ đọc D3, nghiệm thu, đóng băng SHA-256) rồi `python src/03b_evaluate_preprocessing.py` (chỉ chạy được khi mã băm khớp); quyết định và kết quả trong `docs/cau_c_decisions.md`.

Câu d: `python src/04_prepare.py` (chỉ chạy khi dữ liệu câu c khớp mã băm đóng băng); quyết định trong `docs/cau_d_decisions.md`.

Câu e: `python src/05_cluster.py` (chỉ chạy khi mã băm câu c, d khớp); kết quả trong `docs/cau_e_results.md`.

Báo cáo: `reports/Bao_cao_giua_ky.pdf` (nguồn LaTeX `reports/latex/`, biên dịch bằng `cd reports/latex && tectonic main.tex`).

Quy tắc: `customer_id` chỉ là khóa; nhãn phân khúc và nhật ký lỗi nằm trong `data/hidden/` và **không** được đưa vào ma trận đặc trưng.
