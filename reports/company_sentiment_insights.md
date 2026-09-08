# Insight cảm xúc doanh nghiệp và ảnh bàn giao

Notebook nguồn: `notebooks/05_company_sentiment_insights.ipynb`

Ngày chạy: 08/09/2026

Dữ liệu: 8.417 review sạch từ `data/processed/reviews_cleaned.xlsx`

## Kết quả toàn bộ dữ liệu

| Cảm xúc | Số review | Tỷ lệ |
| :--- | ---: | ---: |
| Tích cực | 6.208 | 73,76% |
| Trung tính | 1.639 | 19,47% |
| Tiêu cực | 570 | 6,77% |

Ảnh dùng cho báo cáo:

- `reports/figures/wordcloud_positive_all.png`
- `reports/figures/wordcloud_negative_all.png`

## Case study doanh nghiệp

Notebook chỉ chọn doanh nghiệp đạt tối thiểu 50 review và lấy hai doanh nghiệp có nhiều review nhất. Kết quả mang tính mô tả dữ liệu, không phải bảng xếp hạng nơi làm việc.

| Doanh nghiệp | Review | Rating TB | Tích cực | Trung tính | Tiêu cực |
| :--- | ---: | ---: | ---: | ---: | ---: |
| FPT Software | 2.014 | 3,68/5 | 57,75% | 33,37% | 8,89% |
| NashTech | 308 | 3,87/5 | 69,16% | 26,30% | 4,55% |

Ảnh dùng cho báo cáo:

- `reports/figures/company_fpt_software_sentiment_distribution.png`
- `reports/figures/wordcloud_fpt_software_positive.png`
- `reports/figures/wordcloud_fpt_software_negative.png`
- `reports/figures/company_nashtech_sentiment_distribution.png`
- `reports/figures/wordcloud_nashtech_positive.png`
- `reports/figures/wordcloud_nashtech_negative.png`

## Demo web

Ứng dụng Streamlit có ba khu vực: tổng quan dữ liệu, dashboard insight theo doanh nghiệp và dự đoán cảm xúc bằng model thật. Dashboard áp dụng ngưỡng mẫu trước khi so sánh doanh nghiệp và chỉ tạo WordCloud khi người dùng mở phần tương ứng để giảm thời gian tải.

Chạy cục bộ:

```powershell
.\.venv311\Scripts\python.exe -m streamlit run app.py
```

## Lưu ý diễn giải

Nhãn cảm xúc trong dashboard là weak label suy ra từ rating. Các tỷ lệ và WordCloud mô tả nội dung trong tập review hiện có, không chứng minh quan hệ nhân quả và không nên dùng riêng lẻ để kết luận chất lượng doanh nghiệp.
