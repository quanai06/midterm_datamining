# Checklist yêu cầu đề thi (không được bỏ sót ý nào)

Tham số: N = 200+20a | thuộc tính ≥ 8+b | outlier ≥ 5+2c (xem `src/params.py`)

## Câu a – Thiết kế & sinh dữ liệu (1,5đ)
- [x] Đề xuất thuộc tính: ý nghĩa, kiểu dữ liệu, lý do chọn (bảng data dictionary)
- [x] Số bản ghi đúng N; số thuộc tính đầu vào ≥ 8+b (có cả số + phân loại)
- [x] Cơ chế sinh dữ liệu giả lập (cụm ẩn, phân phối, tương quan giữa thuộc tính)
- [x] Giải thích giả định và hạn chế của cơ chế sinh
- [x] Giá trị thiếu ở ≥ 2 thuộc tính (nêu cơ chế MCAR/MAR/MNAR và tỉ lệ)
- [x] Nhiễu ngẫu nhiên ở ≥ 2 thuộc tính số (nêu loại nhiễu, tham số)
- [x] Ngoại lai ≥ 5+2c bản ghi (nêu cách tạo)
- [x] Lưu dữ liệu TRƯỚC (`data/01_generated/data_clean_reference.csv`) và SAU (`data/01_generated/data_raw_corrupted.csv`) khi thêm nhiễu/ngoại lai
(Đã làm xong trong code và docs, kiểm tra bằng `python src/01_generate_data.py`: 42/42 đạt. Phần viết vào báo cáo DOCX làm ở cuối.)

## Câu b – Data Understanding (2,5đ)
- [x] Phân bố từng thuộc tính (hist/box, skewness)
- [x] Quan hệ giữa thuộc tính (tương quan, scatter, categorical vs numeric)
- [x] Giá trị thiếu (tỉ lệ, mẫu thiếu, cơ chế nghi ngờ)
- [x] Nhiễu và ngoại lai (phát hiện bằng IQR/z-score/…)
- [x] ≥ 3 đặc điểm/vấn đề quan trọng + ảnh hưởng dự kiến đến phân cụm
(Đã làm: `src/02_data_understanding.py`, kết luận ở `docs/cau_b_findings.md`, hình ở `figures/02_understanding/`. Phần viết vào báo cáo DOCX làm ở cuối.)

## Câu c – Data Preprocessing (2,5đ)
- [x] Xử lý giá trị thiếu (KHÔNG khôi phục từ dữ liệu sạch)
- [x] Xử lý nhiễu
- [x] Xử lý ngoại lai (KHÔNG mặc định loại bỏ hết: cân nhắc giữ/winsorize/cờ hóa)
- [x] Giải thích cơ sở lựa chọn
- [x] Phân tích thay đổi dữ liệu sau xử lý (trước/sau; dùng dữ liệu sạch CHỈ để đối chiếu)
(Đã làm: `src/03_preprocess.py`, `src/03b_evaluate_preprocessing.py`; quyết định, kết quả, hạn chế ở `docs/cau_c_decisions.md`. Nghiệm thu 15/15; dữ liệu đóng băng SHA-256. Viết báo cáo DOCX làm ở cuối.)

## Câu d – Data Preparation (2đ)
- [x] Mã hóa, chuẩn hóa, chọn thuộc tính / giảm chiều / giảm đối tượng (chọn phù hợp)
- [x] Giải thích cơ sở lựa chọn
- [x] Phân tích thay đổi dữ liệu sau khi áp dụng
(Đã làm: `src/04_prepare.py`; quyết định, bằng chứng, hạn chế ở `docs/cau_d_decisions.md`. Nghiệm thu 13/13. Chờ sinh viên review.)

## Câu e – Phân cụm & đánh giá (1,5đ)
- [x] So sánh: dữ liệu nhiễu/outlier (biến đổi tối thiểu để chạy) vs dữ liệu sau xử lý + chuẩn bị
- [x] Giải thích chọn thuật toán, thiết lập thí nghiệm, chỉ số đánh giá
- [x] Phân tích đặc điểm các cụm và sự thay đổi kết quả
- [x] Nêu rõ nếu số đối tượng / không gian đặc trưng thay đổi; so sánh công bằng
- [x] Độ nhạy từ câu c: `preferred_category` (unknown / KNN-7 / mode / bỏ cột, file `variants/`) và có/không có 34 dòng cực trị hợp lệ (`preprocessing_flags.csv`)
(Đã làm: `src/05_cluster.py`; thiết lập, kết quả, phân tích và hạn chế ở `docs/cau_e_results.md`. Cần ghi: số dòng 260 ở mọi nhánh chính, số cột 20/20/21/19, N = 226 ở biến thể bỏ cực trị.)

## Sản phẩm nộp
- [ ] 1. Báo cáo PDF/DOCX (`reports/`)
- [ ] 2. Dữ liệu ban đầu + sau nhiễu/outlier (`data/01_generated/data_clean_reference.csv`, `data/01_generated/data_raw_corrupted.csv`)
- [ ] 3. Dữ liệu sau tiền xử lý và chuẩn bị (`data/02_preprocessed`, `data/03_prepared`)
- [ ] 4. Mã nguồn/notebook chạy lại được (`src/`, `notebooks/`, `logs/`)
- [ ] 5. Lịch sử hội thoại AI + khai báo công cụ (`docs/ai_log.md`)

## Tiêu chí chấm năng lực cộng tác AI (mỗi câu)
(i) xác định & phân tích vấn đề · (ii) tư duy phản biện & kiểm chứng · (iii) sáng tạo trong thiết kế/lựa chọn ·
(iv) điều chỉnh & cải thiện giải pháp · (v) nhất quán giữa hội thoại và sản phẩm
→ Ghi lại trong `docs/ai_log.md` những chỗ AI sai/được sửa, phương án đã loại, lần lặp cải thiện.
