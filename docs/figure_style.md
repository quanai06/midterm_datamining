# Style biểu đồ

Nguồn quy ước: repo `ChenLiu-1996/figures4papers` (đọc qua `scientific-figure-making/SKILL.md`, `references/api.md`, `references/design-theory.md`).
Lưu ý: nội dung repo được lấy qua công cụ tóm tắt, không phải bản chép nguyên văn; các giá trị cụ thể (mã màu, cỡ chữ) cần đối chiếu lại với repo gốc nếu muốn trích dẫn chính xác.

Áp dụng (xem `src/viz_style.py`): bỏ spine trên/phải; legend không khung; font Helvetica/Arial/DejaVu Sans; nét 2pt; lưới tối giản;
`tight_layout(pad=2)`; xuất PNG + PDF 300 dpi; chữ dạng vector (`svg.fonttype='none'`, `pdf.fonttype=42`); hatch làm mã hóa phụ để đọc được khi in đen trắng;
nhãn panel (a), (b)...; họ màu xanh - đỏ - trung tính.

Sai khác có chủ đích:
| Nội dung | Repo | Ở đây | Lý do |
|---|---|---|---|
| Xanh chính | `#0F4D92` | `#3775BA` | bộ kiểm tra màu (skill dataviz, `validate_palette.js`) báo trượt dải sáng: quá tối so với các màu còn lại |
| Xanh lam-lục | `#42949E` | `#2E9C8A` | trượt ngưỡng sắc độ (đọc như màu xám); cùng họ màu |
| Thứ tự màu | | xanh, đỏ, lam-lục, tím | chỉ thứ tự này đạt mọi kiểm tra ở các cặp liền kề (CVD dE 9,8; vùng nhìn bình thường dE >= 15) |
| Cỡ chữ / nét | 15-16 / 2,5 | 13 / 1,8 | hình lưới nhiều panel khổ lớn (tới 22 x 8 inch) |
| Viền cột | đen dày | trắng mảnh | mật độ cột cao; hatch dùng để phân biệt |

Bộ 4 màu `#3775BA, #B64342, #2E9C8A, #9A4D8E` đạt: dải sáng, sắc độ, CVD, vùng nhìn bình thường (cặp liền kề), tương phản >= 3:1 trên nền trắng.
Chỉ dùng cho tối đa 4 chuỗi; với `--pairs all` bộ này chưa đạt ngưỡng 15 ở cặp đỏ-tím, nên các hình có 4 nhóm đều có thêm ký hiệu hình dạng (o, s, ^, D) hoặc hatch.
Các hình heatmap dùng thang tuần tự một họ xanh hoặc thang phân kỳ xanh - đỏ với điểm giữa xám.
