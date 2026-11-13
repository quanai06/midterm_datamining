# Câu d - Data Preparation: quyết định, bằng chứng, kết quả, hạn chế

Mã: `src/04_prepare.py` (+ `src/prep_utils.py`). Đầu vào: dữ liệu câu c đã đóng băng (script kiểm tra mã băm trước khi chạy). Không đọc D0-D2, nhãn ẩn.
Chạy lại: `python src/04_prepare.py` (~6 s, CPU). Log: `logs/04_prepare.log`. Nghiệm thu 13/13.
**Ghi chú quy trình:** các lựa chọn dưới đây do AI đề xuất và triển khai ngay theo yêu cầu "làm nhanh"; sinh viên chưa xem đề xuất trước khi chạy. Cần bác bỏ/chỉnh ở lượt review cuối nếu không đồng ý
(mọi lựa chọn thay thế đã có sẵn dưới dạng biến thể).

## 1. Quyết định chính (ma trận `X_main`, 260 x 19) và cơ sở
| Bước | Quyết định | Bằng chứng (chỉ từ dữ liệu đã xử lý) | Thay thế |
|---|---|---|---|
| Loại metadata | Bỏ `customer_id` và mọi cờ/nhật ký khỏi ma trận; id lưu riêng `customer_ids.csv` (cùng thứ tự) | Nghiệm thu: không cột metadata trong X | |
| Biến số lệch phải | `log1p` cho 6 cột (n_orders, AOV, recency, sessions, minutes, pages), rồi StandardScaler; `age`, `discount`, `abandon` chỉ z-score | Quy tắc câu b/c (skew > 1 sau khi loại ô vi phạm, không phải biến tỉ lệ bị chặn); sessions skew 1,16 sát ngưỡng | Không log (xem scaler) |
| Scaler | **log1p + StandardScaler** | Bảng mục 2 | MinMax, Robust |
| Thứ tự `membership_tier` | Mã 0/1/2 (bronze < silver < gold) rồi z-score, một cột | Hạng có thứ tự thật; one-hot sẽ bỏ thông tin thứ tự và thêm 2 cột | One-hot |
| Nhị phân `app_installed` | z-score, một cột | E[d²] = 2 như mọi cột chuẩn hóa | |
| Định danh `device_type` (3 mức), `preferred_category` (5 mức) | One-hot **không** chuẩn hóa theo phương sai, nhân `w = 1/sqrt(1 - Σp²)`: w = 1,301 (device), 1,125 (category, tính trên 247 nhãn đã biết) | Mục 3 | z-score cả dummy; dummy 0/1 thô |
| `unknown` của category | Hàng toàn số 0 ở khối category (cách đều mọi nhãn, không tạo trục riêng cho 13 dòng) | z-score của dummy "unknown" sẽ là 4,36 | Cột `unknown` riêng (`X_cat_unknown_col`) |
| Chọn thuộc tính | Bỏ `total_spend_12m` | R² của log(total) theo log(n_orders), log(AOV) = 0,987; VIF 82,5. Bỏ xong: VIF n_orders 67,1 -> 7,3; AOV 14,8 -> 2,3. Các cột còn lại R² <= 0,90 và có ý nghĩa nghiệp vụ riêng (pages VIF 10,2 giữ nguyên) | Giữ (`X_full_features`) |
| Giảm chiều | **Không PCA** trong ma trận chính | Giữ 90% phương sai cần 9/19 PC nhưng chỉ bảo toàn 74% lân cận 5-NN; 5 PC chỉ 55%. N = 260 và 19 cột nên không cần giảm chiều để tính | PCA 90% (`X_pca90`), lưu cùng bản không giảm chiều |

## 2. StandardScaler, MinMaxScaler, RobustScaler trong clustering dựa khoảng cách (10 cột số, số đo trên dữ liệu thật)
| Biểu diễn | 5 dòng giữ bao nhiêu % tổng bình phương | cột lớn nhất giữ | max |giá trị| | ô vượt 4 sd |
|---|---|---|---|---|
| thô + Standard | 21,9% | 11,3% | 12,7 | 17 |
| thô + MinMax | 7,7% | 30,7% | 1,0 | 17 |
| thô + Robust | **80,7%** | 52,3% | **85,2** | 17 |
| log1p + **Standard** | 6,6% | 11,9% | 5,8 | 2 |
| log1p + MinMax | 4,6% | 24,7% | 1,0 | 2 |
| log1p + Robust | 8,8% | 16,9% | 5,5 | 2 |
- **Standard** (trung bình 0, sd 1): mỗi cột đóng góp bằng nhau vào khoảng cách Euclid, đúng điều K-Means (cực tiểu hóa tổng bình phương) giả định; nhạy với cực trị nếu không log (thô: 17 ô vượt 4 sd).
- **MinMax** (khoảng [0,1]): khoảng tính từ min/max nên một cực trị quyết định thang của cả cột; dữ liệu có cực trị hợp lệ được giữ lại nên nén phần lớn khách về một đoạn hẹp (cột lớn nhất chiếm 24,7-30,7%).
- **Robust** (trung vị/IQR): bền với cực trị khi tính thang nhưng **không chặn** cực trị: trên dữ liệu thô một khách "cá voi" ở cách 85 đơn vị IQR, 5 dòng chiếm 80,7% tổng bình phương. Sau log thì gần Standard (8,8% so với 6,6%) nhưng các cột hai đỉnh (discount, abandon) có IQR rộng nên thang không cân bằng (cột lớn nhất 16,9%).
- Chọn **log1p + Standard**: cân bằng nhất (cột lớn nhất 11,9% ~ 1/10) và ít bị vài dòng chi phối nhất trong nhóm có phương sai bằng nhau. Robust sau log là phương án thay thế gần tương đương (không lưu riêng).

## 3. One-hot không được phép chi phối khoảng cách (tỉ trọng phương sai theo khối)
| Mã hóa | numeric (9) | tier+app (2) | device | category |
|---|---|---|---|---|
| A. z-score cả dummy | 45,0% | 10,0% | 15,0% | **30,0%** |
| B. dummy 0/1 thô | 72,9% | 16,2% | 4,8% | 6,2% |
| **C. dummy x w (chính)** | 69,4% | 15,4% | 7,7% | 7,4% |
- A: khối category chiếm 30% dù chỉ là một biến trên 13, và dummy "unknown" (5% dòng) bị khuếch đại thành z = 4,36. Loại.
- B: mỗi biến nominal chỉ đóng góp ít hơn một cột số. Loại.
- C: mỗi biến số/hạng/nhị phân/device đóng góp đúng 7,71% khoảng cách bình phương, category 7,43% (E[d²] = 1,925 vì 13 dòng `unknown` cách đều); nghiệm thu tự kiểm tra điều này.
- Chênh lệch giữa các cách xử lý `unknown`/biến thể category nhỏ về hình học (5-NN trùng 0,94-0,97, tương quan hạng khoảng cách >= 0,9995) nhưng **bỏ hẳn category** làm 31% lân cận 5-NN đổi (0,685), nên đây là biến đáng được thử ở câu e.

## 4. K-Means hay K-Prototypes/Gower?
- Dữ liệu hỗn hợp nhưng thiên về số: 9 cột số + 1 hạng + 1 nhị phân + 2 nominal (13 thuộc tính). Quan hệ của biến phân loại với biến số yếu (câu b: Cramér V <= 0,24; epsilon² của device <= 0,11, của app <= 0,07).
- **K-Means trên X_main** là phương án chính: với trọng số `w` và z-score, mỗi biến đóng góp cùng mức vào khoảng cách, tức là cùng điều kiện mà tham số gamma của K-Prototypes phải điều chỉnh; gamma không có giá trị "đúng" trong học không giám sát. Hạn chế: tâm cụm của biến one-hot là tỉ lệ (không phải mức thật) và K-Means giả định cụm gần cầu cùng cỡ.
- **K-Prototypes: chưa chạy được**, môi trường không có thư viện `kmodes` (không cài thêm khi chưa được yêu cầu).
- **Gower** (khoảng chuẩn hóa theo miền cho số và hạng, 0/1 cho nhị phân và nominal, bỏ qua biến khi `unknown`): ma trận 260 x 260 lưu tại `variants/gower_distance.npy`, dùng được cho phân cấp liên kết trung bình hoặc DBSCAN với `metric="precomputed"`.
  Hình học khác X_main rõ: 5-NN trùng 0,71, tương quan hạng khoảng cách 0,894.

## 5. Kết quả (`reports/tables/d_*.csv`, `figures/04_preparation/`)
- 14 thuộc tính -> 13 (bỏ total_spend) -> **19 cột** (9 số + tier + app + 3 device + 5 category); 260 hàng; không NaN/inf; 11 cột z-score có trung bình 0, sd 1.
- Skew sau biến đổi: n_orders 3,33 -> 0,52; sessions 1,16 -> -0,63; minutes 2,07 -> 0,47; pages 1,64 -> 0,52; recency 1,67 -> -0,55; **AOV 10,92 -> 1,67 (max z = 5,8, vì cực trị hợp lệ còn nguyên)**.
- Tỉ trọng khoảng cách trước chuẩn bị (số thô): total_spend 99,78%, các cột còn lại gộp 0,22%; sau chuẩn bị mỗi biến ~7,7%.
- Pipeline sklearn (`pipeline/prep_pipeline.joblib`) tái tạo `X_main.csv` với sai khác 4,4e-16 khi nạp lại trong tiến trình mới; tái dựng tay bằng NumPy khớp pipeline.

## 6. File
`data/03_prepared/`: `X_main.csv`, `customer_ids.csv`, `pipeline/` (`prep_pipeline.joblib`, `pca90.joblib`, `feature_metadata.csv`, `prep_config.json`), `variants/`
(`X_full_features`, `X_pca90`, `X_cat_unknown_col`, `X_cat_knn7`, `X_cat_mode`, `X_no_category`, `gower_distance.npy`), `PREPARED_SHA256.json`. Hình: figD1-figD5.

## 7. Hạn chế
- w và các tham số chuẩn hóa được tính trên chính 260 dòng (không có tách train/test vì không giám sát); khi có dữ liệu mới cần dùng lại tham số đã lưu.
- Cực trị hợp lệ vẫn còn trong log-space (AOV max z = 5,8) và 80 khách (30,8%) có n_orders = 1 tạo một khối điểm trùng giá trị sau log; K-Means dễ gom khối này.
- Đa cộng tuyến còn lại: pages_per_session VIF 10,2, n_orders 7,3. Không loại thêm vì chưa có căn cứ nghiệp vụ là hàm của biến khác.
- `unknown` = hàng toàn số 0 là quy ước (cách đều mọi nhãn), không phải thông tin; chưa đo ảnh hưởng đến cụm (việc của câu e).
- Gower dùng log cho cột lệch để cùng thang với X_main, và không có phép đối sánh độc lập về chất lượng.
