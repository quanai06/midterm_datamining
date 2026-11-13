# Câu c - Data Preprocessing: nhật ký quyết định, kết quả và hạn chế

Mã: `src/03_preprocess.py` (xử lý, chỉ đọc D3 + data dictionary), `src/03b_evaluate_preprocessing.py` (đối chiếu D0 sau đóng băng),
`src/dm_utils.py` (hàm dùng chung). Log: `logs/03_preprocess.log`, `logs/03b_evaluate_preprocessing.log`.
Chạy lại: `python src/03_preprocess.py` (~30 s, CPU) rồi `python src/03b_evaluate_preprocessing.py`.

## 1. Kiểm soát rò rỉ
- `03_preprocess.py` chỉ gọi `read_input()` cho `data_raw_corrupted.csv` và `data_dictionary.csv` (chốt chặn đường dẫn lúc chạy). Nghiệm thu quét mã nguồn: không chứa tên D0-D2, nhật ký chèn lỗi hay thư mục dữ liệu ẩn.
- Sau khi 15/15 kiểm tra nghiệm thu đạt, 4 file kết quả được ghi SHA-256 vào `data/02_preprocessed/FROZEN_SHA256.json` và chmod 444. `03b` từ chối chạy nếu bất kỳ mã băm nào lệch
  (đã thử: sửa 1 byte, script từ chối; đã khôi phục đúng bản đóng băng).
- Refactor `dm_utils.py` làm riêng và kiểm thử trước: 35/35 file kết quả câu b (bảng, JSON, log, PNG) giống hệt trước khi refactor.
- `data_raw_corrupted.csv` không đổi (SHA-256 trước/sau bằng nhau, ghi trong manifest).

## 2. Quyết định và cơ sở
| Vấn đề | Quyết định | Cơ sở (chỉ từ D3) | Phương án thay thế đã loại |
|---|---|---|---|
| 7 ô không hợp lệ | Giữ dòng; vô hiệu hóa **từng ô** (6 vi phạm luật miền + 1 vi phạm đẳng thức) rồi điền | Mỗi dòng chỉ một ô hỏng; các ô còn lại qua mọi kiểm tra | Xóa 7 dòng (mất 13 ô hợp lệ/dòng); cắt ngưỡng (tạo điểm dồn) |
| C0062 (AOV 3093 vs total/n = 40,04) | Ô AOV bị vô hiệu hóa và **điền bằng mô hình như mọi ô khác** (kết quả 40,75). total/n chỉ là kiểm tra hợp lý | Data Dictionary ghi `total_spend ~ n_orders x AOV` (xấp xỉ), AOV không nêu phạm vi thời gian, và câu b đo sd bền của độ lệch ở dòng bình thường là 0,164 (~16%) nên đẳng thức không chính xác. Điều kiện bạn đặt (từ điển xác nhận đẳng thức chính xác) **không thỏa**, nên không gán cứng total/n | Gán AOV = total/n: sẽ biến xấp xỉ thành đẳng thức |
| Ô nghi trong ba ô của đẳng thức | Chọn ô có phân vị biên cực đoan nhất: AOV 99,6% vs total 60,8% và n_orders 59,2%; hạng bronze (trung vị total 79) khớp với total = 360 | Quy tắc tổng quát trong mã, không viết cứng C0062 | |
| Điền khuyết số | Theo thí nghiệm che ô: 15% ô quan sát, 50 lần, **cùng mask cho mọi phương pháp**, so sánh ghép cặp, quy tắc 1-SE (chọn phương pháp đơn giản nhất có MAE <= MAE tốt nhất + SE của nó; thứ tự đơn giản: median < median theo thiết bị < KNN5 < KNN10 < iterative) | Bảng mục 3 | Trung vị (tạo điểm dồn tại một giá trị); xóa dòng (mất 23,5%) |
| `preferred_category` | Giữ mức `unknown` (13 ô); các biến thể KNN-7 và mode chỉ lưu ở file riêng để thử ở câu e | Mục 4: bất định cao; không dùng ngưỡng accuracy chung | Điền KNN-7 hoặc mode |
| Nhiễu | **Không khử từng giá trị.** Không làm mượt, không winsorize, không cắt ngưỡng | Nhiễu độc lập từng ô không phục hồi được từ D3; nhìn thấy duy nhất qua độ lệch đẳng thức (sd bền 0,164, không tách được phần của từng cột). Làm mượt trộn khách khác nhóm và co các hồ sơ hiếm | Làm mượt/binning; hòa giải AOV với total/n cho mọi dòng (chỉ có thể làm kiểm tra sau, không dùng ở quy trình chính) |
| Cực trị hợp lệ | Chỉ đánh dấu (`extreme_tukey3_valid`: 34 dòng, 56 ô vượt 3 IQR và không vi phạm luật nào); không sửa, không xóa | Tukey 1,5 IQR gắn cờ 94 dòng (precision 0,14 ở câu b), không dùng làm tiêu chí xóa | Winsorize hoặc xóa |
| Điểm dồn ở biên | Giữ nguyên, ghi cờ. Không ô nào được điền rơi đúng giá trị biên | sessions = 0,5 (12 dòng), recency = 0 (15), recency = 365 (11), n_orders = 1 (80), age = 18 (22), đều thấy ngay trong D3 | Jitter hoặc điền lại: bịa dữ liệu |
| Cờ và nhật ký | File riêng: `preprocessing_flags.csv`, `preprocessing_audit.csv`; ma trận đặc trưng đúng id + 14 cột | | |
| Ranh giới câu c/d | Đầu ra giữ đơn vị và độ chính xác gốc: không log, không chuẩn hóa, không mã hóa. Log chỉ dùng bên trong bộ điền khuyết | | |

## 3. Chọn phương pháp điền: kết quả thí nghiệm che ô (mỗi hàng: MAE trung bình, độ lệch chuẩn qua 50 lần, SE; cùng 50 mask cho mọi phương pháp)
| Cột | Phương pháp (đơn vị gốc) | Được chọn |
|---|---|---|
| age (36 ô/lần; **năm**) | median 8,82 (SD 0,84, SE 0,118); median theo thiết bị 8,13; KNN5 7,10; **KNN10 6,89 (SD 0,86, SE 0,122)**; iterative 7,20 | KNN10: các phương pháp còn lại đều vượt ngưỡng 1-SE (0,4683 trên thang chọn) |
| avg_session_minutes (34 ô/lần; **phút**) | median 7,69 (SD 1,41, SE 0,200); KNN5 4,10; KNN10 4,00; **iterative 3,04 (SD 0,60, SE 0,085)** | iterative |
| avg_order_value | median 44,4 USD; KNN10 34,4; **iterative 15,8** | iterative |
| recency_days | median 65,8 ngày; KNN10 48,6; **iterative 43,7** | iterative |
| discount_usage_rate | median 0,281; **KNN5 0,123 (trong 1-SE của KNN10 0,122)**; iterative 0,187 | KNN5 (đơn giản hơn) |
| sessions_per_month | median 6,35; **KNN10 3,78**; iterative 3,85 (cũng trong 1-SE) | KNN10 (đứng trước iterative theo thứ tự đơn giản) |
| cart_abandon_rate | median 0,195; **KNN5 0,068 (trong 1-SE của KNN10 0,067)**; iterative 0,084 | KNN5 |
Chọn dựa trên MAE ở thang lens (log1p + median/IQR cho cột lệch, median/IQR cho cột còn lại; age tuyến tính nên thứ hạng giống hệt đơn vị năm). Đầy đủ SD, SE,
chênh ghép cặp so với phương pháp tốt nhất: `reports/tables/c_imputer_selection_numeric.csv`. Đính chính pilot: nhãn "thang log" cho age ở bản đề xuất là sai
(age chỉ chuẩn hóa theo IQR); báo cáo bằng năm.

## 4. preferred_category: vì sao giữ `unknown` (dựa trên bất định, không dùng ngưỡng chung)
- Che 15% ô quan sát, 50 lần, cùng mask: độ chính xác mode toàn cục 0,254 (SE 0,010), mode theo thiết bị 0,218, mode theo hạng 0,197, **KNN-7 0,402 (SE 0,010)**.
  KNN-7 hơn mode toàn cục 0,148 (SE ghép cặp 0,013), tức tốt hơn thật, nhưng vẫn sai khoảng 60% lần.
- KNN-7 trên chính 13 ô thiếu: xác suất lớp dự đoán trung bình 0,528 (từ 0,397 đến 0,710), entropy chuẩn hóa 0,638 (0 = chắc chắn, 1 = đều 5 lớp): mô hình không tự tin.
  Số nhãn dự kiến sai nếu điền: khoảng 7,8/13.
- Lập luận: khác biệt có ý nghĩa thống kê giữa các phương pháp **không** đủ để biện minh cho việc gán nhãn khi mỗi nhãn gán vào vẫn có xác suất sai lớn hơn xác suất đúng
  và biến này chiếm 5% dòng. `unknown` không bịa thông tin; chi phí là thêm một mức hiếm khi mã hóa ở câu d.
- Ảnh hưởng đến phân cụm **chưa kết luận ở đây**: cần thử ở câu e (xem `docs/requirements_checklist.md`): bốn phương án `unknown` / KNN-7 / mode / bỏ cột, so sánh độ ổn định và chỉ số cụm.

## 5. Kết quả xử lý (chỉ dựa trên D3 và dữ liệu đã xử lý)
- 260 dòng giữ nguyên; **72 ô thay đổi**: 7 ô không hợp lệ -> điền, 21 ô thiếu của age -> điền, 31 ô thiếu của avg_session_minutes -> điền, 13 ô category -> `unknown`. Mọi ô khác giống D3 từng giá trị.
- Ô C0062 AOV: điền 40,75 so với total/n = 40,04 (chênh 1,8%, chỉ để kiểm tra).
- Tỉ trọng bình phương khoảng cách của 7 dòng đã sửa: z-score thô 0,314 -> 0,041; log+IQR 0,152 -> 0,044. 5-NN max/median: 19,1 -> 10,5 (z-score thô). Hopkins: 0,940 -> 0,885 (z-score thô), 0,835 -> 0,754 (log+IQR).
  max |Pearson - Spearman|: 0,80 -> 0,52. sd/sd bền: sessions 7,7 -> 1,0; avg_session_minutes 11,1 -> 1,4; avg_order_value 10,6 -> 8,1; total_spend 9,3 -> 9,3 (không đổi).
- Điểm dồn ở biên không đổi (12, 15, 11, 80, 22); không ô điền nào rơi đúng giá trị biên. Độ lặp của giá trị điền không vượt mức bốc ngẫu nhiên từ phân phối quan sát (bootstrap p95: age 4 <= 5; minutes 2 <= 3).
- Giá trị điền ít phân tán hơn giá trị quan sát: sd(điền)/sd(quan sát) = 0,57 (age), 0,90 (avg_session_minutes).

## 6. Đối chiếu với D0 (sau đóng băng; số đo tổng hợp, không phải khôi phục từng ô)
(A) Sai khác phân phối cả cột (`c_eval_distribution_vs_clean.csv`):
- sessions: std/std(D0) 8,31 -> 1,10; Wasserstein 4,33 -> 0,53. avg_session_minutes: 9,07 -> 1,03; Wasserstein 6,70 -> 0,48. age: 1,24 -> 0,97, skew 3,34 -> 0,43 (D0 0,41).
- **Không cải thiện hoặc xấu đi:** KS của sessions giữ 0,050 (do điểm dồn ở 0,5); KS của age 0,011 -> 0,023; avg_order_value skew 9,19 -> 10,92 (D0 1,76) và std/std(D0) 7,5 -> 5,6 vì các giá trị
  cực trị hợp lệ còn nguyên; total_spend, n_orders, pages_per_session không đổi (không có ô nào bị chạm). Wasserstein của recency 1,19 -> 1,38 (xấu nhẹ, xem ô C0242 dưới đây).

(B) Sai số tổng hợp tại ô đã thay đổi, đo bằng D0 (`c_eval_cell_error_aggregate.csv`):
- age, 21 ô thiếu: MAE 7,14 năm (mốc trung vị toàn cục 8,24; ước lượng che ô 6,89); tương quan với D0 0,58; sd(điền)/sd(D0) 0,53. Cải thiện khiêm tốn, phù hợp pilot "tuổi gần như không đoán được".
- avg_session_minutes, 31 ô thiếu: MAE 2,95 phút so với D0 (mốc trung vị 6,23); sàn nhiễu |D2 - D0| = 2,15; tương quan 0,92. So với giá trị quan sát trước khi xóa (D2): MAE 3,79,
  **cao hơn ước lượng che ô 3,04** (+25%).
- 7 ô không hợp lệ (mỗi cột 1 ô, chỉ mô tả): sai số so với D0 là age 3 năm, AOV 6,2 USD, discount 0,049, abandon 0,027, sessions 3,8, minutes 0,7. **Ngoại lệ xấu: recency của C0242
  được điền 210 ngày, D0 là 69 (sai 141 ngày, tệ hơn cả mốc trung vị toàn cục: 26).**

(C) Phục hồi cấu trúc (`c_eval_structure_vs_clean.csv`), Before (D3) -> After:
- mean |Δ Pearson| so với D0: 0,178 -> 0,063 (max 0,857 -> 0,481); mean |Δ Spearman|: 0,020 -> 0,017.
- 5-NN giữ lại so với D0: 0,527 -> 0,630 (z-score thô); 0,572 -> 0,622 (log+IQR). Spearman khoảng cách: 0,772 -> 0,878; 0,861 -> 0,937.
- Hopkins (D0: 0,833 / 0,724): 0,940 -> 0,885 và 0,835 -> 0,754. Silhouette theo nhãn ẩn (chẩn đoán; D0: 0,298 / 0,327): 0,281 -> 0,288 và 0,288 -> 0,314.
- Còn cách D0 đáng kể: 5-NN chỉ giữ 0,62-0,63. Nhiễu chưa được xử lý (ở câu b, riêng nhiễu D0 -> D1 đã giữ 0,79 / 0,67), cộng với cực trị hợp lệ và giá trị điền.

(D) preferred_category, 13 ô thiếu: nhãn KNN-7 đúng 3/13 (0,23; Wilson 95%: 0,08-0,50) và mode đúng 3/13: không phân biệt được ở mức n = 13, và không trái với quyết định giữ `unknown`.
Phân bố nhãn KNN-7 (fashion 68, electronics 54, home 50, beauty 53, grocery 35) gần phân bố D0 (70, 54, 48, 53, 35) hơn mode (fashion 80), dù độ chính xác từng ô như nhau.

## 7. Hạn chế
- Luật miền (ví dụ sessions <= 100, minutes <= 240) là giả định của người phân tích; ô hỏng nằm trong miền hợp lệ không bị bắt. Giả định mỗi dòng chỉ hỏng một ô.
- Thí nghiệm che ô dùng mask ngẫu nhiên (MCAR) trong khi thiếu của avg_session_minutes có dấu hiệu theo thiết bị; ước lượng 3,04 phút lạc quan hơn thực tế đo được 3,79 (n = 31, một mẫu).
  Các lần che chồng lấn nhau nên SE chỉ phản ánh biến thiên theo mask, không phải sai số tổng quát hóa. **Tôi không đổi pipeline sau khi biết kết quả D0** để tránh rò rỉ.
- Điền đơn trị đánh giá thấp bất định và co phương sai (age 0,53-0,57). Điền dựa trên các cột khác rồi phân cụm trên chính các cột đó có thể làm cụm trông chặt hơn.
- Chưa xử lý nhiễu: 4 cột nhiễu (SNR 20-28) và hai tác dụng phụ của việc chặn biên (sessions = 0,5; recency = 0) còn nguyên.
- Cực trị hợp lệ còn nguyên làm sd của avg_order_value, total_spend, n_orders vẫn cao hơn D0; xử lý ảnh hưởng khoảng cách là việc của câu d.
- C0242 recency điền sai nhiều (xem trên); tác động tới phân cụm chưa đo.
- Dữ liệu giả lập; D0 là "sự thật" duy nhất nhờ vào dữ liệu giả lập, không có tương đương trong dữ liệu thật.

## 8. Tệp tạo ra
`data/02_preprocessed/`: `data_after_preprocessing.csv`, `preprocessing_audit.csv` (72 dòng: trước, sau, phương pháp, lý do), `preprocessing_flags.csv`,
`variants/preferred_category_variants.csv`, `FROZEN_SHA256.json`. Bảng: `reports/tables/c_*.csv`, `c_eval_*.csv`. Hình: `figures/03_preprocessing/` (7 hình Before/After, chỉ D3),
`figures/03_preprocessing_eval/` (4 hình đối chiếu D0).
