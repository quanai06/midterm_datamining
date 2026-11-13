# Data Dictionary — Câu a (đã duyệt và triển khai)

> **Trạng thái:** bảng thuộc tính dưới đây là thiết kế đã duyệt. Số liệu "phân tích khả năng phân biệt" ở giữa tài liệu
> là của **bản mô phỏng thử** (trước khi cài đặt). Số liệu thật của bản đã triển khai nằm ở mục cuối
> "Kết quả triển khai". Bản đã sinh: `data/01_generated/data_dictionary.csv` (kèm tỉ lệ thiếu, min/median/max thực tế).

MSSV 24022433 → a=3, b=3, c=4 → N = 260, thuộc tính ≥ 11, outlier ≥ 13.
Đề xuất: **14 thuộc tính đầu vào** (10 số + 4 phân loại), **15 outlier**.

**Không thuộc tập đặc trưng:** `customer_id` (định danh), `true_segment` (nhãn ẩn, chỉ dùng đối chiếu ở câu e),
`is_bridge` (cờ khách "cầu nối", chỉ dùng phân tích). Ba cột này lưu ở file metadata riêng, pipeline không đọc.

## Bảng thuộc tính

| # | Thuộc tính | Nhóm | Ý nghĩa | Kiểu | Miền giá trị | Phân phối dự kiến | Lý do chọn |
|---|---|---|---|---|---|---|---|
| 1 | age | Nhân khẩu | Tuổi khách | int | 18–70 | Chuẩn theo phân khúc, cắt biên | Đặc điểm khách hàng; phân biệt yếu (nhiễu hợp lý, kiểm tra thuộc tính ít thông tin) |
| 2 | n_orders_12m | Mua hàng | Số đơn 12 tháng gần nhất | int | ≥ 1 (thực tế 1–60) | Nhị thức âm, lệch phải | Tần suất giao dịch |
| 3 | avg_order_value | Chi tiêu | Giá trị đơn trung bình (USD) | float | > 0 (thực tế 8–300) | Lognormal, lệch phải | Quy mô mỗi lần mua; tách khách mua đắt và khách mua rẻ |
| 4 | total_spend_12m | Chi tiêu | Tổng chi 12 tháng (USD) | float | > 0 | Lệch phải mạnh (skew ≈ 4,4) | Giá trị khách hàng; **cố ý** tương quan ≈ 0,91 với n_orders (đa cộng tuyến cho câu b, d) |
| 5 | recency_days | Mua hàng | Số ngày từ đơn gần nhất | int | 0–365 | Gamma, lệch phải | Mức độ còn "sống"; phân biệt khách không thường xuyên |
| 6 | discount_usage_rate | Khuyến mãi | Tỉ lệ đơn dùng mã/giảm giá | float | [0, 1] | Beta | Định nghĩa nhóm săn khuyến mãi |
| 7 | sessions_per_month | Tương tác | Số phiên truy cập/tháng | float | ≥ 0 | Lognormal | Mức độ gắn bó với nền tảng |
| 8 | avg_session_minutes | Tương tác | Thời lượng phiên trung bình (phút) | float | 1–60 | Lognormal | Mức độ tương tác |
| 9 | pages_per_session | Duyệt | Số trang sản phẩm xem/phiên | float | 1–40 | Lognormal | Hành vi duyệt; tách nhóm "chủ yếu duyệt" |
| 10 | cart_abandon_rate | Duyệt | Tỉ lệ giỏ hàng bị bỏ | float | [0, 1] | Beta/chuẩn cắt | Duyệt nhiều nhưng không chốt đơn |
| 11 | membership_tier | Phân loại thứ tự | Hạng thành viên | ordinal | bronze < silver < gold | Rời rạc theo phân khúc | Gắn với giá trị khách; cần mã hóa thứ tự (câu d) |
| 12 | device_type | Phân loại định danh | Thiết bị chính | nominal | mobile, desktop, tablet | Rời rạc | Hành vi theo thiết bị; dùng cơ chế thiếu MAR |
| 13 | preferred_category | Phân loại định danh | Ngành hàng ưa thích | nominal | fashion, electronics, home, beauty, grocery | Rời rạc | Sở thích mua; cần one-hot (câu d) |
| 14 | app_installed | Nhị phân | Đã cài ứng dụng | binary | 0/1 | Bernoulli | Mức gắn bó kênh |

## Bốn phân khúc ẩn (n = 48 / 78 / 66 / 68)

| Phân khúc | n_orders | avg_order_value | recency | discount | sessions | pages/phiên | abandon |
|---|---|---|---|---|---|---|---|
| HighValue (giá trị cao) | ~24 | ~85 | ~12 | ~0,12 | ~14 | ~9 | ~0,25 |
| PromoHunter (săn khuyến mãi) | ~14 | ~35 | ~30 | ~0,78 | ~9 | ~7 | ~0,45 |
| Infrequent (ít mua) | ~3 | ~50 | ~150 | ~0,25 | ~2 | ~4 | ~0,35 |
| Browser (chủ yếu duyệt) | ~2 | ~40 | ~60 | ~0,15 | ~18 | ~16 | ~0,80 |

Cơ chế tạo chồng lấn (đã hiệu chỉnh, xem mục dưới): 18% khách là "cầu nối" (tham số trộn 35–65% giữa phân khúc của họ và một
phân khúc khác), phân tán trong cụm vừa phải, một nhân tố "hoạt động" tiềm ẩn chung tạo tương quan trong cụm, xác suất
các biến phân loại được làm mềm 35% về phân phối đều.

## Phân tích khả năng phân biệt (bằng chứng: `src/design_validation/`)

Đây là kiểm tra thiết kế của bộ sinh dữ liệu, dùng nhãn ẩn chỉ để chẩn đoán. Không thuộc pipeline phân cụm.

**Vòng 1 (tham số ban đầu):** K-Means k=4 trên dữ liệu sạch đạt ARI = 0,98, LDA phân biệt từng cặp 99–100%. Bộ dữ liệu quá
dễ: câu e sẽ gặp hiệu ứng trần, và không thực tế. → **Bác bỏ, thêm chồng lấn.**

**Vòng 2 (cấu hình chốt, 10 seed sinh dữ liệu):** K-Means k=4: ARI = 0,75 ± 0,03, NMI = 0,71, Silhouette = 0,15.

Cặp phân khúc khó nhất và đặc trưng phân biệt chính (seed 42, LDA 5-fold):

| Cặp | Độ chính xác | Đặc trưng phân biệt nhất (Cohen d) |
|---|---|---|
| HighValue vs PromoHunter | 0,90 | discount_usage_rate (3,2), avg_order_value (1,7), total_spend (1,3) |
| HighValue vs Infrequent | 0,95 | total_spend (2,5), sessions (2,3), n_orders (2,3) |
| HighValue vs Browser | 0,97 | cart_abandon (3,4), total_spend (3,3), n_orders (3,1) |
| PromoHunter vs Infrequent | 0,94 | discount (2,5), sessions (1,9), n_orders (1,9) |
| PromoHunter vs Browser | 0,97 | discount (3,5), n_orders (2,8), cart_abandon (2,3) |
| Infrequent vs Browser | 0,95 | cart_abandon (2,9), sessions (2,6), pages (2,3) |

Đặc trưng nào nuôi nhóm nào: HighValue ← total_spend, avg_order_value; PromoHunter ← discount_usage_rate;
Browser ← sessions, pages, cart_abandon; Infrequent ← recency, n_orders thấp. Mỗi nhóm có ít nhất một "chữ ký" riêng.

## Phát hiện quan trọng từ kiểm tra (cần ghi vào báo cáo)

1. **Trần ARI bị khách "cầu nối" quyết định.** ARI trên khách không phải cầu nối = 0,93, trên khách cầu nối = 0,15.
   Sau xử lý, ARI ≈ 0,75 là mức tối đa thực tế, không phải ≈ 1.
2. **Silhouette thấp (0,15) và chọn k mơ hồ:** k=3 cho 0,148, k=4 cho 0,145 (seed 42). Ở câu e phải dùng nhiều chỉ số
   (Silhouette, Davies-Bouldin, Calinski-Harabasz, độ ổn định) và báo cáo trung thực nếu silhouette nghiêng về k=3.
3. **Đa cộng tuyến đúng thiết kế:** n_orders ~ total_spend r = 0,91; sessions ~ pages r = 0,66.
4. **Tính điểm yếu có chủ đích:** age và avg_order_value giải thích phương sai phân khúc thấp (η² = 0,25).
5. **Hạn chế về cách hiệu chỉnh:** tham số chồng lấn được chọn để dữ liệu sạch có ARI ≈ 0,75. Đây là lựa chọn thiết kế phải
   công khai trong báo cáo, và không được chỉnh lại để làm kết quả tiền xử lý đẹp hơn.

## Dự kiến cho bước sinh dữ liệu (sau khi duyệt)

- Giá trị thiếu: age (MCAR ~8%), avg_session_minutes (MAR theo device_type ~12%), preferred_category (MCAR ~5%).
- Nhiễu: avg_order_value (nhân lognormal), recency_days và avg_session_minutes (cộng Gaussian).
- Outlier: 15 dòng sửa tại chỗ (8 cực trị hợp lệ + 7 lỗi nhập liệu), ghi trong `injection_log.csv`.
- `01_clean` đủ 260 dòng không thiếu; `02_noisy` chứa thiếu + nhiễu + outlier.
- Hạn chế dữ liệu giả lập: cụm do chính ta đặt, không có yếu tố thời gian, phân phối chọn tay, giả định mỗi khách có ≥ 1 đơn/năm.

## Kết quả triển khai (`src/01_generate_data.py`, seed 42)

**Chuỗi file:** `data_clean_reference.csv` (D0) → `data_after_noise.csv` (D1) → `data_after_outliers.csv` (D2) →
`data_raw_corrupted.csv` (D3). Thông tin ẩn trong `data/hidden/`: `ground_truth.csv` (phân khúc, cờ cầu nối, đặc điểm ẩn),
`injection_log.csv` (1.040 ô nhiễu, 42 ô ngoại lai của 15 khách, 65 ô thiếu).

**Kiểm tra nghiệm thu:** 42/42 đạt (log: `logs/01_generate_data.log`). Chạy lại cho md5 từng file giống hệt, kể cả giữa
tiến trình khác nhau và thư mục trống (mô phỏng Colab).

| Ràng buộc đề | Kết quả |
|---|---|
| N = 200 + 20a | 260 |
| Thuộc tính >= 8 + b | 14 (>= 11) |
| Thiếu ở >= 2 thuộc tính | 3: age 21 ô, avg_session_minutes 31 ô, preferred_category 13 ô |
| Nhiễu ở >= 2 thuộc tính số | 4: avg_order_value, recency_days, avg_session_minutes, sessions_per_month |
| Ngoại lai >= 5 + 2c | 15 khách khác nhau (8 hợp lệ, 7 lỗi) |

**Tham số hiệu chỉnh cuối (chỉ dựa trên D0):** khoảng cách phân khúc quanh trung bình chung x1,0 (E) và x1,5 (I, W, S);
phân tán trong cụm x0,6; nhiễu phương trình x0,75; cat_soft = 0,20; conv0 = -2,6. Các cấu hình đã thử và lý do loại bỏ nằm
trong `docs/ai_log.md` (mục 3-4) và `src/design_validation/calibrate_overlap.py`.

**Độ tách phân khúc trên D0:** ARI K-Means (k=4) = 0,73 (seed chính), 0,71 ± 0,05 trên 10 seed sinh mới;
Silhouette = 0,23; cây quyết định một thuộc tính tốt nhất = 0,68 (pages_per_session), sessions_per_month = 0,64.

**Sai khác so với thiết kế đã duyệt (cần nêu trong báo cáo):**
1. Bỏ phụ thuộc "discount giảm theo hạng gold" vì tạo vòng lặp (hạng suy từ total_spend).
2. Ngoại lai "đại lý" và "cá voi" đặt cường độ theo hàng rào Tukey và ràng buộc <= 1 đơn/phiên, thay cho hệ số nhân cố định.
3. Điều kiện ARI sạch được kiểm bằng trung bình 10 seed sinh mới (không dùng khi hiệu chỉnh) trong 0,70-0,80, cộng dung sai
   0,60-0,85 cho seed chính.

**Hạn chế còn lại (trung thực):**
- ARI trung bình 0,71 sát mép dưới dải mục tiêu; độ chính xác một thuộc tính 0,68 sát ngưỡng 0,70.
- Khách Browser có trung vị n_orders = 1 (sàn "ít nhất 1 đơn"): nhiều khách dồn tại sàn.
- Đuôi tự nhiên của HighValue có khách tới ~100 đơn/năm; kết hợp ngoại lai "cá voi" cho tổng chi tới hàng chục nghìn USD.
- Silhouette thấp (0,23) và chọn k có thể mơ hồ; câu e phải dùng nhiều chỉ số.
- 99-101 dòng vượt hàng rào Tukey ở D2 (gồm đuôi tự nhiên lệch phải): bộ phát hiện sẽ cho nhiều dương tính giả.
- Toàn bộ là dữ liệu giả lập: cụm do ta định nghĩa nên ARI đo mức khôi phục thiết kế của ta.
