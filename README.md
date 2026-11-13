# Phân nhóm khách hàng thương mại điện tử

**Báo cáo:** [`reports/Bao_cao_giua_ky.pdf`](reports/Bao_cao_giua_ky.pdf) · nguồn LaTeX: [`reports/latex/`](reports/latex/) · **Nhật ký cộng tác AI:** [`docs/ai_log.md`](docs/ai_log.md)

## Bài toán

Một sàn thương mại điện tử có tập khách hàng với các đặc điểm (tuổi, thiết bị, hạng thành viên, ngành hàng ưa thích) và hành vi
(lịch sử mua hàng, chi tiêu, mức dùng khuyến mãi, hành vi duyệt, mức tương tác). Doanh nghiệp muốn chia khách thành vài nhóm có hành vi
giống nhau bên trong và khác nhau giữa các nhóm, để chăm sóc, giữ chân và tiếp thị phù hợp (khách giá trị cao, săn khuyến mãi,
mua không thường xuyên, chủ yếu duyệt). Không có nhãn nhóm sẵn có, nên đây là bài toán **phân cụm** (học không giám sát).

Khó khăn nằm ở chất lượng dữ liệu: nhiễu đo lường, ngoại lai (lỗi nhập liệu khác với khách cực trị có thật), giá trị thiếu, phân phối lệch phải,
thuộc tính trùng lặp, dữ liệu hỗn hợp số/thứ tự/định danh với thang đo rất khác nhau. Những vấn đề này làm méo khoảng cách giữa các khách,
mà thuật toán phân cụm dựa vào chính khoảng cách đó. Dự án trả lời: **hiểu, tiền xử lý và chuẩn bị dữ liệu thay đổi chất lượng và độ ổn định của phân cụm
đến mức nào so với việc chỉ biến đổi tối thiểu để thuật toán chạy được.**

Vì không có dữ liệu thật, dự án dùng bộ dữ liệu giả lập 260 khách, 14 thuộc tính, có bốn nhóm ẩn chồng lấn, nhiễu, 15 khách có ngoại lai
(8 cực trị hợp lệ, 7 lỗi) và 3 cột thiếu giá trị. Nhóm ẩn **chỉ dùng để đánh giá**, không dùng để chọn thuộc tính, tham số hay số cụm.

## Quy trình

| Bước | Nội dung | Script | Đầu ra chính |
|---|---|---|---|
| 1. Sinh dữ liệu | Mô hình nhân quả nhiều tầng; thêm nhiễu, ngoại lai, giá trị thiếu theo chuỗi D0 → D3 | `src/01_generate_data.py` | `data/01_generated/`, `data/hidden/` |
| 2. Hiểu dữ liệu | Phân phối, thiếu, tương quan, ngoại lai, tác động lên khoảng cách; phần "mù" chỉ đọc D3 | `src/02_data_understanding.py` | `figures/02_understanding/`, `docs/cau_b_findings.md` |
| 3. Tiền xử lý | Sửa từng ô không hợp lệ, điền khuyết theo thí nghiệm che ô, giữ cực trị hợp lệ; đóng băng SHA-256 | `src/03_preprocess.py`, `src/03b_evaluate_preprocessing.py` | `data/02_preprocessed/`, `docs/cau_c_decisions.md` |
| 4. Chuẩn bị dữ liệu | log1p + z-score, one-hot có trọng số, bỏ thuộc tính dư thừa; bộ biến đổi sklearn | `src/04_prepare.py` | `data/03_prepared/`, `docs/cau_d_decisions.md` |
| 5. Phân cụm và đánh giá | K-Means và phân cấp trên khoảng cách Gower; so sánh trước/sau xử lý | `src/05_cluster.py` | `data/04_clustering/`, `docs/cau_e_results.md` |

## Kết quả chính

K-Means trên cùng 260 khách; ARI so với nhóm ẩn (khoảng tin cậy 95% bằng bootstrap):

| | Trước xử lý (chỉ biến đổi tối thiểu + z-score) | Sau tiền xử lý và chuẩn bị |
|---|---|---|
| ARI tại k = 4 | 0,611 [0,515; 0,708] | **0,745** [0,658; 0,833] |
| k chọn bằng Silhouette | 7 (ARI 0,434) | **4** (ARI 0,745) |
| Ổn định giữa các lần chạy (ARI) | 0,685 | 0,958 |

Mốc trần (dữ liệu sạch qua cùng pipeline): ARI 0,777. Tại k = 4, đóng góp riêng của bước chuẩn bị (so với chỉ tiền xử lý) chưa chứng minh được
(+0,044, khoảng tin cậy chứa 0); bằng chứng nằm ở việc chọn đúng k và độ ổn định. Chi tiết, các kết quả không thuận lợi và hình trong báo cáo.

## Cấu trúc thư mục

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── 01_generate_data.py              # sinh D0-D3, 42 kiểm tra nghiệm thu
│   ├── 02_data_understanding.py         # hiểu dữ liệu (Phần A "mù" chỉ đọc D3, Phần B đối chiếu)
│   ├── 03_preprocess.py                 # tiền xử lý, 15 kiểm tra nghiệm thu, đóng băng SHA-256
│   ├── 03b_evaluate_preprocessing.py    # đối chiếu với dữ liệu sạch, chỉ chạy khi mã băm khớp
│   ├── 04_prepare.py                    # chuẩn bị dữ liệu, 13 kiểm tra nghiệm thu
│   ├── 05_cluster.py                    # phân cụm và đánh giá
│   ├── dm_utils.py, prep_utils.py       # hàm dùng chung
│   ├── viz_style.py                     # phong cách biểu đồ
│   └── design_validation/               # mã hiệu chỉnh độ khó của dữ liệu giả lập
├── data/
│   ├── 01_generated/                    # D0 sạch, D1 +nhiễu, D2 +ngoại lai, D3 +thiếu (đầu vào của các bước sau), data_dictionary.csv
│   ├── hidden/                          # nhóm ẩn và nhật ký chèn lỗi (chỉ dùng để đánh giá)
│   ├── 02_preprocessed/                 # dữ liệu sau tiền xử lý, nhật ký từng ô thay đổi, cờ, biến thể, FROZEN_SHA256.json
│   ├── 03_prepared/                     # X_main.csv (260x19), pipeline/ (bộ biến đổi, siêu dữ liệu), variants/
│   └── 04_clustering/                   # nhãn cụm, tâm cụm, cấu hình chạy, mã băm kết quả
├── notebooks/
│   └── 01_generate_data_colab.ipynb     # chạy bước sinh dữ liệu trên Colab
├── figures/                             # hình theo từng bước (PNG + PDF)
│   ├── 02_understanding/  03_preprocessing/  03_preprocessing_eval/  04_preparation/  05_clustering/
├── reports/
│   ├── Bao_cao_giua_ky.pdf              # báo cáo
│   ├── latex/                           # nguồn LaTeX (main.tex, sec*.tex, app_*.tex)
│   └── tables/                          # bảng số liệu của từng bước (a_*, b_*, c_*, d_*, e_*)
├── docs/
│   ├── ai_log.md                        # nhật ký cộng tác AI: quyết định, sai sót và chỉnh sửa
│   ├── data_dictionary.md               # thiết kế thuộc tính và cơ chế sinh
│   ├── cau_b_findings.md  cau_c_decisions.md  cau_d_decisions.md  cau_e_results.md   # kết quả và quyết định từng bước
│   ├── figure_style.md                  # quy ước biểu đồ
│   └── requirements_checklist.md
└── logs/                                # log chạy từng script
```

## Cài đặt và chạy lại

Yêu cầu: Python 3.10+; CPU là đủ (toàn bộ quy trình chạy trong vài phút, không cần GPU).

```bash
pip install -r requirements.txt
python src/01_generate_data.py            # sinh dữ liệu và kiểm tra nghiệm thu
python src/02_data_understanding.py
python src/03_preprocess.py               # tiền xử lý và đóng băng kết quả
python src/03b_evaluate_preprocessing.py  # chỉ chạy được khi mã băm khớp bản đã đóng băng
python src/04_prepare.py
python src/05_cluster.py
cd reports/latex && tectonic main.tex     # biên dịch báo cáo (XeTeX)
```

## Tái lập và kiểm soát rò rỉ

- Hạt giống cố định (42); chạy lại cho kết quả giống hệt (đã kiểm tra bằng mã băm các file đầu ra).
- Nhóm ẩn và nhật ký chèn lỗi nằm trong `data/hidden/`, không có mặt trong bất kỳ file dữ liệu nào đưa vào phân tích.
- Sau bước tiền xử lý, các file kết quả được ghi SHA-256 (`data/02_preprocessed/FROZEN_SHA256.json`) và khóa ghi; các bước sau từ chối chạy nếu mã băm lệch.
  Bước đối chiếu với dữ liệu sạch chỉ chạy sau khi đã đóng băng và không dùng dữ liệu sạch để thay thế từng giá trị.
- Mọi kết quả không thuận lợi và mọi lần sửa phương pháp sau khi thấy kết quả được ghi trong [`docs/ai_log.md`](docs/ai_log.md).
