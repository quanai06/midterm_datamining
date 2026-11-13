# Câu e - Phân cụm và đánh giá: thiết lập, kết quả, phân tích, hạn chế

Mã: `src/05_cluster.py` (chạy lại ~12 s, CPU; kết quả giống hệt: md5 `labels.csv` trùng giữa các lần chạy). Log: `logs/05_cluster.log`.
Đầu vào bị khóa: mã băm câu c và câu d phải khớp bản đóng băng, nếu không script từ chối. Nhãn: `data/04_clustering/labels.csv`; tâm cụm `centers_*.csv`;
cấu hình, hạt giống, mã băm đầu vào: `run_config.json`; mã băm kết quả: `RESULTS_SHA256.json`. Bảng: `reports/tables/e_*.csv`. Hình: `figures/05_clustering/figE1-E7`.

## 1. Thiết lập và cơ sở
**Thuật toán.** K-Means (`n_init = 50`, `random_state = 42`) là thuật toán chính: khoảng cách Euclid trên ma trận đã chuẩn hóa là điều câu d đã chuẩn bị (mỗi biến đóng góp ngang nhau),
nhanh, tái lập được, tâm cụm đọc được. Thuật toán thứ hai: phân cấp trên khoảng cách Gower (không cần ma trận số, dùng thang riêng), để kiểm tra kết luận không chỉ đúng với K-Means.
Chưa dùng DBSCAN/GMM; K-Prototypes không chạy được (không có thư viện `kmodes`).

**Các nhánh so sánh** (cùng 260 dòng ở mọi nhánh chính; số cột khác nhau và là một phần của sự khác biệt cần đọc):
| Nhánh | Dữ liệu và biến đổi | Số cột |
|---|---|---|
| B0 | D3 + biến đổi tối thiểu (điền trung vị, one-hot, mã hạng), **không** chuẩn hóa | 20 |
| **B1 (Before)** | B0 + z-score mọi cột (nền so sánh công bằng; một người dùng K-Means thông thường sẽ làm vậy) | 20 |
| L2 | dữ liệu sau câu c + đúng biến đổi của B1 ('unknown' là một mức) | 21 |
| **L3 (After)** | `X_main` của câu d (log1p, z-score, one-hot x w, bỏ total_spend) | 19 |
| REF | D0 sạch qua đúng pipeline câu d: **mốc trần, không phải một phương pháp** | 19 |
B0 chỉ để cho thấy tác động của thang đo; L2 tách riêng đóng góp của câu c khỏi câu d.

**Chọn k.** Chỉ bằng chỉ số nội tại (Silhouette cao nhất, k = 2..8). Nhãn phân khúc ẩn không tham gia chọn k, tham số hay biến.
Vì 4 là số phân khúc do người thiết kế đặt, tôi báo cáo cả hai: **k chọn theo nội tại** và **k = 4 cố định** (để so cùng k). Cần nhớ: k = 4 dùng hiểu biết về thiết kế.

**Chỉ số.** Ngoài: ARI, NMI, purity so với phân khúc ẩn (đo độ khôi phục thiết kế, không phải chân lý của dữ liệu thật). Nội tại: Silhouette, Davies-Bouldin, Calinski-Harabasz ở thang riêng
**và ở thang chung** `X_main` (chỉ thang chung so sánh được giữa các nhánh). Ổn định: ARI giữa cụm trên mẫu con 80% (30 lần) và cụm đầy đủ; ARI giữa 20 lần chạy `n_init = 1` và lần tốt nhất.
Khoảng tin cậy: bootstrap 1000 lần theo dòng, ghép cặp, cho cả ARI và chênh lệch ARI.

## 2. Kết quả chính (K-Means)
**Chọn k bằng Silhouette** (`e_k_selection.csv`): B0 k = 2 (Silhouette 0,948, cụm 257/3), B1 k = 7 (0,180; tăng từ 0,111 ở k = 2 đến k = 7 rồi giảm còn 0,163 ở k = 8), L2 k = 7 (0,198), **L3 k = 4 (0,299, đỉnh rõ)**, REF k = 4 (0,307).
Với k chọn theo nội tại, B1 và L2 cho k = 7 và một cụm chỉ gồm 1-2 dòng; L3 chọn đúng 4 cụm mà không cần biết trước.

| k = 4 | ARI [CI 95%] | NMI | purity | Sil. thang chung | DB thang chung | kích thước cụm | ổn định mẫu con 80% | ARI giữa chạy n_init = 1 (trung bình / thấp nhất) |
|---|---|---|---|---|---|---|---|---|
| B0 | 0,035 [0,016-0,061] | 0,130 | 0,358 | 0,130 | 1,445 | 245/12/2/1 | 0,914 ± 0,184 | 1,00 / 1,00 |
| **B1** | 0,611 [0,515-0,708] | 0,571 | 0,835 | 0,254 | 1,449 | 82/68/65/45 | 0,829 ± 0,150 | 0,685 / 0,435 |
| L2 | 0,701 [0,610-0,787] | 0,664 | 0,877 | 0,274 | 1,377 | 82/68/67/43 | 0,949 ± 0,039 | 0,823 / 0,485 |
| **L3** | 0,745 [0,658-0,833] | 0,698 | 0,896 | 0,299 | 1,277 | 77/74/63/46 | 0,970 ± 0,017 | 0,958 / 0,721 |
| REF | 0,777 [0,693-0,856] | 0,737 | 0,912 | 0,295 | 1,293 | 75/74/66/45 | 0,987 ± 0,013 | 0,996 / 0,980 |

| k chọn theo Silhouette | k | ARI [CI] | NMI |
|---|---|---|---|
| B0 | 2 | 0,005 [0,000-0,014] | 0,028 |
| B1 | 7 | 0,434 [0,350-0,530] | 0,432 |
| L2 | 7 | 0,551 [0,463-0,645] | 0,545 |
| L3 | 4 | 0,745 [0,658-0,833] | 0,698 |

**Chênh lệch ARI (bootstrap ghép cặp, `e_bootstrap_ari.csv`):**
| So sánh | k = 4 | k chọn theo nội tại |
|---|---|---|
| L3 - B1 (câu c + d) | **+0,134 [0,058; 0,206]**, dương ở 100% mẫu | **+0,311 [0,224; 0,393]** |
| L2 - B1 (riêng câu c) | **+0,090 [0,040; 0,143]** | +0,117 [0,057; 0,177] |
| L3 - L2 (riêng câu d) | +0,044 [-0,024; 0,115] (**không phân biệt được với 0**; 89% dương) | **+0,194 [0,101; 0,281]** |
| REF - L3 (khoảng cách tới trần) | +0,032 [-0,007; 0,071] (**không phân biệt được với 0**) | cùng giá trị |

**Thuật toán thứ hai: phân cấp trên Gower** (k = 4, `e_gower_hierarchical.csv`; liên kết chọn trước là `average`, kết quả rất yếu nên tôi thêm `complete` và chọn giữa hai loại **chỉ bằng Silhouette Gower**; bảng in cả hai):
| Nhánh | average: ARI (kích thước cụm) | complete: ARI | liên kết được Silhouette Gower chọn: ARI |
|---|---|---|---|
| B1 (Gower trên D3 thô) | 0,047 (238/18/3/1) | 0,126 | average: 0,047 |
| L2 (Gower trên đơn vị gốc đã xử lý) | 0,170 (215/42/2/1) | 0,262 | average: 0,170 |
| L3 (Gower của câu d) | 0,487 (119/71/68/2) | 0,605 | complete: 0,605 |
Thứ tự B1 < L2 < L3 giống K-Means, nhưng mức tuyệt đối thấp hơn nhiều: phân cấp với liên kết trung bình dễ tạo cụm 1-3 dòng trên dữ liệu có cực trị, và Silhouette Gower **chọn nhầm liên kết** ở hai nhánh thô.

## 3. Phân tích sự thay đổi
1. **Thang đo là tác nhân lớn nhất.** Không chuẩn hóa (B0), `total_spend` chi phối nên K-Means tạo cụm 245/12/2/1 (ARI 0,035), trong khi Silhouette ở thang riêng lại là 0,948 (k = 2).
   **Silhouette ở thang riêng không dùng được để so các nhánh**; chỉ thang chung, chỉ số ngoài và độ ổn định mới so sánh công bằng.
2. **Tiền xử lý (câu c) có lợi rõ nhưng khiêm tốn ở k = 4:** +0,090 ARI (CI loại trừ 0), và ổn định tăng (mẫu con 0,829 -> 0,949; chạy lặp 0,685 -> 0,823). Tác động lớn hơn khi k do chỉ số nội tại chọn (ARI 0,434 -> 0,551).
3. **Chuẩn bị (câu d) có lợi chủ yếu ở độ vững chứ không phải ARI tại k = 4:** +0,044 ARI không đủ bằng chứng ở n = 260. Bằng chứng mạnh hơn nằm ở (a) chọn k: L2 chọn 7 còn L3 chọn đúng 4 (ARI +0,194 [0,101; 0,281] ở k chọn theo nội tại);
   (b) ổn định: ARI giữa các lần chạy n_init = 1 từ 0,823 lên 0,958 (thấp nhất 0,485 lên 0,721), tức K-Means trên L3 ít mắc tối ưu cục bộ hơn.
4. **Khách "cầu nối" quyết định trần** (`e_bridge_customers.csv`): trên 213 khách thường, ARI B1 0,812, L2 0,869, **L3 0,966**, REF 1,000; trên 47 khách cầu nối, mọi nhánh chỉ 0,06-0,19 (kể cả REF 0,19).
   Phần cải thiện nằm ở khách thường; không phương pháp nào phục hồi được khách cầu nối vì chúng do thiết kế nằm giữa hai phân khúc.
5. **Dòng hỏng** (`e_corrupted_rows_in_clusters.csv`): ở k = 4 trên B1, 7 dòng hỏng không lập thành cụm riêng (cụm nhỏ nhất 45 dòng); khi k = 7 (do Silhouette chọn), một cụm chỉ gồm 1 dòng và dòng đó là dòng có ô hỏng. Tác hại chính của ô hỏng
   là gián tiếp: làm méo thang đo và làm Silhouette chọn sai k.
6. **Số dòng và không gian đặc trưng:** 260 dòng ở mọi nhánh chính; số cột 20 / 20 / 21 / 19 do bỏ `total_spend`, mã hóa khác nhau và 'unknown'. Vì vậy tôi so ở thang chung `X_main` và bằng chỉ số ngoài, không so Silhouette thang riêng.

## 4. Đặc điểm các cụm (k = 4, `e_cluster_profile_*.csv`; trung vị theo đơn vị gốc của dữ liệu đã xử lý)
| Cụm After (L3) | n | Đặc điểm | Phân khúc ẩn chiếm đa số |
|---|---|---|---|
| 1. Giá trị cao | 46 | 31,5 đơn, tổng chi 3.093 USD, AOV 110, recency 7 ngày, discount 0,08, hạng gold, mobile (72% cài app) | HighValue 42/46 |
| 2. Mua không thường xuyên | 63 | 2 đơn, tổng chi 114 USD, recency 130 ngày, 2 phiên/tháng, desktop (43% cài app) | Infrequent 54/63 |
| 3. Chủ yếu duyệt | 74 | 1 đơn, 17,5 phiên/tháng, 20,7 trang/phiên, bỏ giỏ 0,89, mobile, electronics (84% cài app) | Browser 66/74 |
| 4. Săn khuyến mãi | 77 | discount 0,92, 17 đơn, AOV 28, tổng chi 450 USD, fashion, mobile | PromoHunter 71/77 |
Before (B1) cho bốn cụm tương tự (giá trị cao 45 dòng: 39 HighValue; ít mua 65: 49 Infrequent; duyệt 68: 61 Browser; khuyến mãi 82: 68 PromoHunter) nhưng **lẫn nhiều hơn**:
10 khách Infrequent rơi vào cụm khuyến mãi, 6 khách HighValue rơi vào cụm ít mua. Đúng cụm: 233/260 (89,6%) ở L3 so với 217/260 (83,5%) ở B1.
Gợi ý nghiệp vụ (suy từ hồ sơ): cụm 1 chăm sóc VIP; cụm 2 kích hoạt lại; cụm 3 tăng chuyển đổi (nhắc giỏ hàng); cụm 4 khuyến mãi có kiểm soát biên lợi nhuận.

## 5. Độ nhạy từ câu c và d (`e_sensitivity.csv`, k = 4, ARI so với phân khúc ẩn)
| Biến thể | n x cột | ARI | chênh so với main [CI] | đồng thuận với phân cụm chính |
|---|---|---|---|---|
| main X_main | 260 x 19 | 0,745 | | 1,000 |
| full_features (còn total_spend) | 260 x 20 | 0,764 | +0,019 [0,000; 0,048] | 0,979 |
| cat 'unknown' thành cột riêng / KNN-7 / mode | 260 x 20 / 19 / 19 | 0,745 / 0,745 / 0,754 | 0 / 0 / +0,009 [0,000; 0,027] | 1,000 / 1,000 / 0,989 |
| no_category | 260 x 14 | 0,771 | +0,026 [-0,009; 0,066] | 0,948 |
| pca90 (9 PC) | 260 x 9 | 0,746 | +0,001 [-0,024; 0,029] | 0,979 |
| bỏ 34 dòng cực trị hợp lệ | **226** x 19 | 0,733 (cùng tập dòng, nhãn của lần chạy đầy đủ: 0,738) | | 0,966 |
Kết luận từ số đo: cách xử lý `unknown` (thành cột, KNN, mode) gần như không ảnh hưởng; bỏ hẳn category đổi 5% nhãn và không làm tệ đi; bỏ `total_spend` hay dùng PCA 90% cũng không đổi đáng kể; loại 34 dòng cực trị hợp lệ không cải thiện
(tức giữ chúng không gây hại ở k = 4). **Câu c đã quyết định giữ `unknown`, câu d giữ category: lựa chọn đó không bị bác bỏ, nhưng category cũng không chứng minh được giá trị trên mẫu này.**
`d_variants_geometry.csv` (câu d) còn cho thấy bỏ category đổi 31% lân cận 5-NN dù cụm K-Means hầu như không đổi: hình học cục bộ nhạy hơn cụm.

## 6. Hạn chế và cách đọc
- Một bộ dữ liệu giả lập với một hạt giống sinh. Nhãn ẩn do chính người thiết kế đặt nên ARI đo mức khôi phục thiết kế, không chứng minh chất lượng với khách hàng thật.
- B1 gồm z-score, nên là nền khá mạnh; so với B0 (chỉ biến đổi tối thiểu thật sự) khoảng cách lớn hơn nhiều (+0,576). Kết luận "câu c + d cải thiện ARI từ 0,611 lên 0,745 ở k = 4" gắn với nền B1.
- Khoảng tin cậy bootstrap coi các dòng độc lập và giữ nguyên nhãn phân cụm; không bao gồm biến thiên do chạy lại phân cụm, nên khoảng thực có thể rộng hơn. Với n = 260, nhiều chênh lệch nhỏ (L3 - L2 ở k = 4, REF - L3, độ nhạy) không phân biệt được với 0.
- k = 4 cố định dùng hiểu biết về thiết kế; kết quả chọn theo nội tại (B1 k = 7) là phép thử trung thực hơn nhưng cũng bất lợi cho B1 vì Silhouette tăng gần như liên tục theo k (đến k = 7) trên dữ liệu thô.
- Thuật toán thứ hai (phân cấp Gower) thay đổi liên kết sau khi thấy kết quả; việc chọn liên kết chỉ dùng chỉ số nội tại nhưng chỉ số này chọn nhầm ở hai nhánh thô: ghi để minh bạch (`docs/ai_log.md`).
- Kết quả đối với B0 cho thấy "ổn định" (ARI 0,91 và chạy lặp 1,00) có thể là dấu hiệu của phân hoạch suy biến, không phải chất lượng.
- Chưa thử DBSCAN, GMM, K-Prototypes; chưa đánh giá trên nhiều hạt giống sinh dữ liệu.
