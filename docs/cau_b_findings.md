# Câu b - Data Understanding: kết quả và phát hiện

Nguồn số liệu: `logs/02_data_understanding.log`, `reports/tables/*.csv`, `reports/tables/b_key_numbers.json`.
Hình: `figures/02_understanding/` (PNG 300 dpi + PDF). Mã: `src/02_data_understanding.py` (chạy lại cho kết quả giống hệt).
Quy trình: **Phần A (mù)** chỉ đọc `data_raw_corrupted.csv` (D3) và `data_dictionary.csv`; **Phần B** mới đọc D0-D2 và `data/hidden/`.

## 1. Tổng quan dữ liệu (D3, Phần A)
- 260 dòng x 14 thuộc tính (+ `customer_id` duy nhất, 0 dòng trùng). 10 thuộc tính số, 1 thứ tự (`membership_tier`), 2 định danh, 1 nhị phân.
- Kiểu đọc được khớp khai báo; ngoại lệ: `age` đọc thành float vì có thiếu. Khoảng giá trị thực tế **vượt miền khai báo** ở nhiều cột
  (age tới 150 so với 18-70; avg_session_minutes tới 1440 so với 1-90; sessions_per_month tới 999 so với <= 60).
- Biến phân loại: bronze 67,7% (176), mobile 54,2% (141), cài app 65,8% (171); preferred_category còn 13 ô thiếu (5,0%).
- Mật độ tại biên: 80 khách (30,8%) có n_orders = 1; 11 khách có recency = 365 (trần).

## 2. Phân phối và thống kê mô tả
| Thuộc tính | trung vị | trung bình | mean/median | skew | std / sd bền (IQR/1,349) | ghi chú |
|---|---|---|---|---|---|---|
| total_spend_12m | 211,7 | 1455 | 6,9 | 8,52 | 9,3 | max 62.272 |
| avg_order_value | 43,8 | 86,9 | 2,0 | 9,19 | 10,6 | max 3.093 |
| n_orders_12m | 5 | 15,5 | 3,1 | 3,33 | 1,9 | max 178 |
| sessions_per_month | 10,05 | 14,9 | 1,5 | 15,7 (1,16 sau khi bỏ ô vi phạm miền) | 7,7 | max 999 |
| avg_session_minutes | 8,5 | 18,9 | 2,2 | 14,8 (2,06 sau khi bỏ ô vi phạm) | 11,1 | max 1440 |
| recency_days | 42 | 81,3 | 1,9 | 1,69 | 1,5 | có giá trị âm (-12) |
| pages_per_session | 7,2 | 10,5 | 1,5 | 1,64 | 1,4 | |
| age | 32 | 33,6 | 1,05 | 3,34 (0,43 sau khi bỏ ô vi phạm) | 1,2 | max 150 |
| discount_usage_rate, cart_abandon_rate | 0,21 / 0,43 | 0,39 / 0,52 | | 0,83 / 0,74 | 0,65 / 0,83 | histogram **hai đỉnh** (fig03) |

Quy tắc chọn cột cần log (skew > 1 sau khi loại ô vi phạm miền, và không phải biến tỉ lệ bị chặn): n_orders, avg_order_value, total_spend,
recency, sessions, avg_session_minutes, pages_per_session (7/10 cột).

## 3. Giá trị thiếu
- age 21 ô (8,1%), avg_session_minutes 31 (11,9%), preferred_category 13 (5,0%). 61/260 dòng (23,5%) có ít nhất một ô thiếu; chỉ 199 dòng đủ dữ liệu.
- Thiếu đồng thời: age & avg_session_minutes 4 dòng (kỳ vọng nếu độc lập 2,5); các cặp còn lại 0. Không thấy mẫu thiếu theo khối.
- Tỉ lệ thiếu của avg_session_minutes theo thiết bị: desktop 21,8% (17/78), tablet 12,2% (5/41), mobile 6,4% (9/141). Kiểm định hoán vị:
  p = 0,0037, **nhưng sau Bonferroni (39 phép thử) p = 0,144, không đạt**.
- preferred_category thiếu liên quan device_type: p hoán vị 0,0015 (số kỳ vọng nhỏ nhất 2,05 nên không dùng χ² tiệm cận), Bonferroni 0,0585.
  Đây là tín hiệu cần thận trọng chứ không phải bằng chứng: 39 phép thử mà một vài p nhỏ là điều có thể xảy ra ngẫu nhiên.
- Ảnh hưởng lên thống kê quan sát được (Phần B): trung vị của avg_session_minutes 8,5 (quan sát) vs 8,5 (đầy đủ D2) vs 8,4 (D0);
  theo thiết bị, trung vị ô thiếu so với ô còn lại: desktop 6,8/7,1; mobile 7,6/8,75; tablet 8,7/10,25.

## 4. Tương quan và cấu trúc đặc trưng
- Spearman lớn nhất: n_orders ~ total_spend 0,92; avg_session_minutes ~ pages_per_session 0,85; n_orders ~ recency -0,75; sessions ~ pages 0,74;
  n_orders ~ cart_abandon -0,73; total_spend ~ recency -0,71; avg_order_value ~ discount -0,59.
- **Pearson thô và Spearman lệch nhau rất nhiều** do các ô hỏng: avg_session_minutes ~ pages Pearson 0,049 vs Spearman 0,848; sessions ~ pages 0,053 vs 0,739.
  (Phần B: ở D0 sạch, hai cặp này có Pearson 0,906 và 0,709.)
- VIF trên hạng: total_spend 23,2 và n_orders 23,1 (đa cộng tuyến), pages 5,6, avg_order_value 4,3.
- Trị riêng của ma trận Spearman: 3 thành phần giữ 80% phương sai, 5 thành phần giữ 90% (trên 10 biến số): số chiều hiệu dụng khoảng một nửa.
- Biến phân loại: Cramér V giữa các biến <= 0,24 (yếu). Epsilon² Kruskal: membership_tier ~ total_spend 0,62 và ~ n_orders 0,52 (hạng suy từ chi tiêu);
  preferred_category ~ avg_order_value 0,40; device_type và app_installed gần như không phân biệt các biến số (epsilon² <= 0,108 và <= 0,065).

## 5. Giá trị bất thường
| Phương pháp (Phần A, mù) | Số dòng gắn cờ |
|---|---|
| Luật miền nghiệp vụ | 6 (age 150, recency -12, discount 1,8, sessions 999, minutes 1440, abandon 1,35) |
| Đẳng thức ln(total / (n x AOV)), ngưỡng 5 sd bền = ±0,82 | 1 (AOV gấp ~100 lần, giá trị 3.093 vẫn trong miền) |
| Tukey 1,5 IQR | 94 (36,2%) |
| Tukey 3 IQR | 38 (14,6%) |
| Robust z > 3,5 (trên log) | 33 |
| LOF (k = 20) | 10 |
- Tukey 1,5 đánh dấu 94/260 dòng chỉ vì phân phối lệch phải; đỉnh là total_spend 39 ô, recency 29 ô (khách ngủ đông, không phải lỗi).
- Tukey **bỏ sót** giá trị 1,8 của discount_usage_rate (hàng rào trên 1,95 vì IQR rộng).
- Độ lệch đẳng thức: sd bền quan sát 0,164 (sd thường 0,311 do một dòng lỗi); phù hợp sd kỳ vọng 0,156 từ nhiễu AOV cộng biến thiên tổng chi.

Kiểm chứng sau với nhật ký (Phần B, 15 khách thật sự bị chèn: 8 hợp lệ + 7 lỗi):
| Phương pháp | gắn cờ | bắt được | precision | recall lỗi (7) | recall hợp lệ (8) |
|---|---|---|---|---|---|
| Luật miền + đẳng thức | 7 | 7 | 1,00 | 7/7 | 0/8 |
| Tukey 1,5 | 94 | 13 | 0,14 | 5/7 | 8/8 |
| Tukey 3 | 38 | 12 | 0,32 | 4/7 | 8/8 |
| Robust z (log) | 33 | 10 | 0,30 | 6/7 | 4/8 |
| LOF 20 | 10 | 9 | 0,90 | 6/7 | 3/8 |

## 6. Tác động của nhiễu, ngoại lai, thiếu (Phần B)
- **Nhiễu (D0 -> D1):** trung bình thay đổi -0,6% / +0,3% / +0,9% / +0,4% (avg_order_value, recency, avg_session_minutes, sessions); sd thay đổi -0,3% đến +2,6%;
  KS 0,03-0,05; tương quan D0-D1 0,977-0,999; SNR (phương sai gốc / phương sai nhiễu) 20-28 (recency 674). Phân phối biên gần như giữ nguyên.
  Tác dụng phụ: 12 khách (4,6%) bị dồn về sàn sessions = 0,5 (D0: 0), và số ô recency = 0 tăng từ 4 lên 13 (do chặn dưới).
- **Ngoại lai (D1 -> D2):** sd tăng x8,2 (sessions), x8,3 (avg_session_minutes), x7,3 (avg_order_value), x1,8 (total_spend); skew của sessions 0,71 -> 15,7,
  avg_session_minutes 2,26 -> 15,7. Pearson giữa các cặp thay đổi tới 0,857 (trung bình |Δ| 0,178), Spearman chỉ tới 0,062 (trung bình 0,020).
- **Thiếu (D2 -> D3):** gần như không đổi thống kê tổng hợp bền (trung vị); tác động chính là mất 23,5% dòng đủ dữ liệu nếu xóa theo dòng.

| Giai đoạn | 5-NN giữ lại so với D0 (z-score thô / log+IQR) | Spearman khoảng cách | Hopkins (z-score thô / log+IQR) | Silhouette nhãn ẩn (z-score thô / log+IQR) |
|---|---|---|---|---|
| D0 | 1,00 / 1,00 | 1,00 / 1,00 | 0,833 / 0,724 | 0,298 / 0,327 |
| D1 (+nhiễu) | 0,79 / 0,67 | 0,99 / 0,98 | 0,837 / 0,718 | 0,294 / 0,317 |
| D2 (+ngoại lai) | 0,55 / 0,62 | 0,78 / 0,87 | 0,939 / 0,835 | 0,281 / 0,297 |
| D3 (+thiếu) | 0,53 / 0,57 | 0,77 / 0,86 | 0,940 / 0,835 | 0,281 / 0,288 |

## 7. Các phát hiện quan trọng và ảnh hưởng dự kiến đến phân cụm
1. **Thang đo và ô hỏng quyết định khoảng cách.** Không chuẩn hóa: total_spend chiếm 99,59% tổng phương sai, các thuộc tính còn lại gộp < 0,5%.
   Sau z-score thô, 7 dòng lỗi (2,7% số dòng) vẫn chiếm 31,4% tổng bình phương khoảng cách của mọi cặp; một ô sessions = 999 kéo sd lên 61,8 so với
   sd bền 8,1, nên z-score nén 259 dòng còn lại về gần hằng số ở cột đó. Với log+IQR thì tỉ lệ này là 15,2%. *Dự kiến:* K-Means/khoảng cách Euclid sẽ bị
   một vài dòng và vài cột chi phối nếu không xử lý lỗi và chuẩn hóa phù hợp.
2. **Phân phối lệch phải nặng và biên cứng.** 7/10 cột cần log; 30,8% khách ở n_orders = 1. Dữ liệu không có dạng cầu, nên tâm cụm (trung bình) bị kéo về đuôi.
   *Dự kiến:* cụm của khách chi tiêu thấp bị nén chặt còn nhóm giá trị cao bị trải rộng, ảnh hưởng mật độ.
3. **Thông tin trùng lặp.** n_orders ~ total_spend (rho 0,92, VIF ~23), sessions ~ pages ~ avg_session_minutes, hạng thành viên suy từ chi tiêu (epsilon² 0,62).
   *Dự kiến:* các trục này bị đếm nhiều lần trong khoảng cách, làm cụm nghiêng theo nhóm biến trùng; số chiều hiệu dụng chỉ 3-5.
4. **Ô hỏng làm sai tương quan tuyến tính.** Pearson thay đổi tới 0,857 nhưng Spearman chỉ 0,062. *Dự kiến:* mọi phương pháp dựa vào cấu trúc tuyến tính
   (PCA trên Pearson, Mahalanobis, GMM) sẽ bị ảnh hưởng nặng hơn phương pháp dựa trên hạng.
5. **Lân cận cục bộ rất nhạy với nhiễu và ngoại lai.** Chỉ riêng nhiễu đã làm mất 21-33% lân cận 5-NN dù KS chỉ 0,03-0,05; thêm ngoại lai còn 53-62%.
   Hopkins tăng từ 0,83 lên 0,94 (z-score thô) khi có ngoại lai, nghĩa là **thống kê này bị phồng khi khung dữ liệu bị kéo giãn** và không đáng tin trên D3.
   Silhouette theo nhãn ẩn giảm 0,327 -> 0,288 (log+IQR). *Dự kiến (giả thuyết, cần kiểm ở câu e):* phương pháp dựa mật độ/lân cận nhạy hơn phương pháp dựa tâm cụm.
6. **Thiếu 23,5% dòng và cơ chế chưa kết luận được.** Bằng chứng thiếu phụ thuộc thiết bị tồn tại nhưng không vượt hiệu chỉnh đa phép thử.
   *Dự kiến:* xóa dòng làm mất gần một phần tư dữ liệu và có thể lệch phân bố thiết bị; điền giá trị cần cân nhắc theo nhóm thiết bị.
7. **Phân biệt ngoại lai hợp lệ và ngoại lai lỗi là điểm then chốt.** 7 dòng lỗi bị bắt hoàn toàn bằng luật miền + đẳng thức (precision 1,00),
   nhưng bộ phát hiện thống kê chung bắt được cả 8 khách cực trị hợp lệ lẫn hàng chục dòng bình thường (Tukey 1,5: precision 0,14). *Dự kiến:* xóa mọi dòng bị gắn cờ sẽ loại
   nhóm khách giá trị cao thật.

## 8. Hạn chế của phân tích này (cần nêu trong báo cáo)
- "Mù" là mù về thủ tục, không phải về tri thức: người phân tích cũng là người thiết kế bộ sinh, nên việc chọn kiểm tra (ví dụ thiếu theo device_type, đẳng thức total ~ n x AOV)
  phản ánh hiểu biết trước. Luật miền (RULES) là giả định nghiệp vụ của người phân tích.
- Đa phép thử: 39 kiểm định cơ chế thiếu; p nhỏ riêng lẻ không nên diễn giải khi chưa hiệu chỉnh.
- Hopkins, kNN và PCA tính trên ma trận điền trung vị tạm thời và hai cách biểu diễn (z-score thô, log+median/IQR); kết quả phụ thuộc cách biểu diễn.
  Các cột lấy log được chọn từ D3 theo quy tắc nêu ở mục 2.
- Silhouette theo nhãn ẩn chỉ để chẩn đoán độ tách, không phải kết quả phân cụm.
- Dữ liệu giả lập; không suy ra kết luận cho khách hàng thật.

## Danh mục hình
fig02 giá trị thiếu | fig03 histogram thô | fig04 histogram log1p | fig05 boxplot | fig06 countplot | fig07 biến phân loại và biến số |
fig08 tương quan Spearman và Pearson | fig09 Cramér V và epsilon² | fig10 quan hệ chính và đẳng thức | fig11 phát hiện ngoại lai |
fig12 khoảng cách, mật độ, PCA | fig13 tác động nhiễu (ECDF) | fig14 tác động ngoại lai | fig15 thay đổi tương quan |
fig16 cấu trúc qua các giai đoạn | fig17 PCA D0 và D3 theo nhãn ẩn (chẩn đoán).
