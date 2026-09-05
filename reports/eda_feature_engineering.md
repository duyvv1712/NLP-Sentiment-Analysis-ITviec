# Báo cáo TV2: Tổng quan dữ liệu, EDA và trích xuất đặc trưng

## 2.1. Tổng quan bộ dữ liệu ITviec Reviews

Bộ dữ liệu sau tiền xử lý tại `data/processed/reviews_cleaned.xlsx` gồm **8.417 review** và **23 trường**, không có dòng trùng hoàn toàn. Dữ liệu bao gồm:

- Thông tin định danh và doanh nghiệp: `id`, `Company Name`, `Cmt_day`.
- Nội dung review: `Title`, `What I liked`, `Suggestions for improvement`.
- Điểm tổng thể: `Rating` từ 1 đến 5 sao.
- Năm điểm khía cạnh: lương và phúc lợi, đào tạo, quản lý, văn hóa, văn phòng.
- Nội dung đã xử lý: `raw_review_text`, `clean_basic_text`, `clean_advance_text`.
- Đặc trưng từ điển: `pos_w`, `neg_w`, `pos_e`, `neg_e`, `total_we`, `sentiment_ratio`.
- Nhãn mục tiêu: `sentiment` gồm `Positive`, `Neutral`, `Negative`; đây là **rating-derived weak labels** được suy ra từ `Rating`, không phải nhãn vàng do con người đọc nội dung và gán.

Dữ liệu khuyết thiếu chỉ xuất hiện ở `What I liked` (1 dòng, 0,01%) và `Suggestions for improvement` (5 dòng, 0,06%). Các cột đã tiền xử lý và cột nhãn không có giá trị khuyết thiếu. Có 6 dòng thuộc ba nhóm `clean_advance_text` trùng; một nhóm có cùng text nhưng khác weak label. Trước khi chia dữ liệu modeling, nhóm bất đồng bị loại toàn bộ và nhóm trùng cùng nhãn chỉ giữ một dòng, còn **8.413 dòng**.

## 2.2. Phân tích khám phá dữ liệu

### Phân bố số sao và nhãn cảm xúc

| Rating | Số review |
|---:|---:|
| 1 | 124 |
| 2 | 446 |
| 3 | 1.639 |
| 4 | 2.698 |
| 5 | 3.510 |

![Phân bố điểm đánh giá](figures/eda_rating_distribution.png)

| Nhãn | Số review | Tỷ lệ |
|---|---:|---:|
| Positive | 6.208 | 73,76% |
| Neutral | 1.639 | 19,47% |
| Negative | 570 | 6,77% |

![Phân bố nhãn cảm xúc](figures/eda_sentiment_counts.png)

Lớp Positive chiếm gần ba phần tư dữ liệu và lớn gấp khoảng 10,9 lần lớp Negative. Vì vậy, Accuracy không đủ để đánh giá mô hình; các thí nghiệm tiếp theo cần ưu tiên **Macro F1** và theo dõi Recall của lớp Negative.

### Độ dài nội dung review

| Trường | Trung bình | Trung vị | Phân vị 95 | Phân vị 99 | Lớn nhất |
|---|---:|---:|---:|---:|---:|
| What I liked | 50,29 | 36 | 131 | 237,36 | 1.400 |
| Suggestions for improvement | 29,97 | 20 | 78 | 184 | 876 |

![Phân bố độ dài nội dung](figures/eda_text_length_distribution.png)

Phân bố độ dài lệch phải rõ rệt: phần lớn review ngắn, nhưng có một số ngoại lệ rất dài. TF-IDF phù hợp với độ dài biến thiên này vì biểu diễn theo trọng số thay vì dùng số lần xuất hiện tuyệt đối.

### Quan hệ giữa điểm khía cạnh và cảm xúc tổng thể

Tương quan Spearman với `Rating`, theo thứ tự giảm dần:

| Khía cạnh | Tương quan Spearman |
|---|---:|
| Management cares about me | 0,7368 |
| Salary & benefits | 0,7343 |
| Culture & fun | 0,6566 |
| Training & learning | 0,6398 |
| Office & workspace | 0,5423 |

![Tương quan giữa các điểm đánh giá](figures/eda_aspect_correlation.png)

Điểm trung bình của mọi khía cạnh đều giảm theo thứ tự Positive → Neutral → Negative. Chênh lệch lớn nhất tập trung ở quản lý, lương/phúc lợi và văn hóa. Đây là các tín hiệu dự báo hữu ích, nhưng **không đưa `Rating` tổng thể vào ma trận đặc trưng** vì nhãn `sentiment` được tạo trực tiếp từ `Rating`; sử dụng trường này sẽ gây rò rỉ nhãn.

![Điểm khía cạnh theo cảm xúc](figures/eda_aspect_by_sentiment.png)

### Phân bố theo công ty và thời gian

Dữ liệu bao phủ **180 công ty** trong giai đoạn 07/2016–05/2025. Phân bố công ty không đồng đều: FPT Software có 2.014 review (23,93%), trong khi 110/180 công ty có dưới 20 review. Insight cấp công ty vì vậy phải hiển thị số mẫu và chỉ kết luận khi đạt ngưỡng tối thiểu.

![Phân bố review theo công ty và thời gian](figures/eda_company_time_distribution.png)

### Chẩn đoán chất lượng weak label và lexicon

- Lexicon ban đầu chỉ có ít nhất một hit trên **12,26%** review. Sau khi TV1 thay bằng thuật toán Greedy Longest Phrase Matching và bổ sung từ điển, độ bao phủ đạt **99,54%** (`total_we` trung bình 0,16 → 6,06; `pos_w` 0,08 → 4,78).
- `pos_e` và `neg_e` bằng 0 trên toàn bộ 8.417 dòng, **kể cả sau bản vá lexicon**, nên hai đặc trưng emoji vẫn là hằng số và không đóng góp gì cho mô hình. Cần TV1 kiểm tra lại khâu đếm emoji.
- `Recommend?` bất đồng với weak label ở nhiều mẫu: 41 Negative vẫn recommend; 411 Neutral và 87 Positive không recommend.
- Những dấu hiệu trên không chứng minh weak label sai, nhưng cho thấy Rating không thể được mô tả là ground truth tuyệt đối.

![Chẩn đoán weak label và lexicon](figures/eda_label_quality_diagnostics.png)

## 3.1. Phương pháp trích xuất đặc trưng

### TF-IDF

TF-IDF được fit trên `clean_advance_text` với cấu hình:

- `max_features=5000`;
- `ngram_range=(1, 2)`;
- `min_df=2`;
- `sublinear_tf=True`.

Sau xử lý text trùng, dữ liệu được chia phân tầng thành **development 80%** và **final test 20%** bằng `random_state=2026`. Final test được khóa và TV2 không tính bất kỳ metric mô hình nào trên tập này. Mọi lựa chọn feature chỉ dùng 5-fold Stratified CV trên development; vectorizer và scaler đều được fit lại bên trong từng fold.

Logistic Regression có `class_weight='balanced'` cho kết quả CV trên development:

| Cấu hình | Macro F1 trung bình | Độ lệch chuẩn |
|---|---:|---:|
| Unigram `(1, 1)` | 0,5396 | 0,0073 |
| Unigram + bigram `(1, 2)` | 0,5579 | 0,0127 |

Cấu hình unigram + bigram cao hơn 0,0183 Macro F1 và được chọn cho bộ đặc trưng text-only bàn giao.

![So sánh n-gram](figures/eda_tfidf_ngram_comparison.png)

### Ablation: mô hình học từ text hay điểm số?

| Nhóm feature | Macro F1 CV | Độ lệch chuẩn | Vai trò |
|---|---:|---:|---|
| Aspect ratings only | 0,7388 | 0,0069 | Diagnostic/tabular upper bound |
| Full structured hybrid | 0,7373 | 0,0101 | Diagnostic |
| Text-only | 0,5579 | 0,0127 | Pipeline NLP dùng cho Web Demo |
| Text + lexicon | 0,5556 | 0,0138 | Lexicon **bản cũ** (bao phủ 12,26%); xem lại ở mục dưới |

Aspect-only cao hơn text-only cho thấy điểm khía cạnh là shortcut rất mạnh đối với weak label tạo từ Rating; riêng kết quả này không chứng minh mô hình hiểu ngôn ngữ.

![Ablation nhóm feature trên development CV](figures/eda_feature_ablation_cv.png)

### Ablation mở rộng: lexicon mới và điểm khía cạnh

Bảng trên chỉ báo Macro F1 nên chưa trả lời được hai câu hỏi nhóm đang cần: điểm khía cạnh có tách được lớp Neutral không, và lexicon sau khi TV1 nâng độ bao phủ lên 99,54% có thực sự cải thiện không.

Thí nghiệm bổ sung (`scripts/run_aspect_hybrid_ablation.py`) dùng lại đúng giao thức — 5-fold Stratified CV trên development, `random_state=2026`, TF-IDF `(1, 2)` 5.000 chiều, `MinMaxScaler` và vectorizer fit lại trong từng fold, Logistic Regression `class_weight='balanced'` — và bổ sung chỉ số từng lớp.

| Nhóm feature | Macro F1 | Neutral F1 | Recall Negative | Negative F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| Text-only | 0,5579 | 0,4456 | 0,4563 | 0,3894 | 0,7158 |
| Text + lexicon | 0,5658 | 0,4508 | 0,4824 | 0,4049 | 0,7204 |
| Text + aspect | 0,7369 | 0,6441 | 0,6952 | 0,6485 | 0,8373 |
| **Text + lexicon + aspect** | **0,7389** | **0,6448** | 0,6995 | **0,6526** | **0,8389** |
| Aspect ratings only | 0,7388 | 0,6353 | **0,7676** | 0,6649 | 0,8337 |

`clean_advance_text` không đổi sau bản vá của TV1 (giống 100% trên cả 8.417 dòng) nên baseline text-only tái lập đúng 0,5579 như bảng cũ; hai bảng vì vậy so sánh trực tiếp được.

**Kiểm định theo cặp fold.** Vì các nhóm feature dùng chung fold và seed, chênh lệch từng fold so sánh trực tiếp được — đây mới là căn cứ kết luận, không phải chênh lệch trung bình:

| So với Text-only | Macro F1 delta | Số fold thắng | Delta nhỏ nhất |
|---|---:|---:|---:|
| Text + lexicon | +0,0079 | 3/5 | −0,0021 |
| Text + aspect | +0,1789 | 5/5 | +0,1635 |
| Text + lexicon + aspect | +0,1809 | 5/5 | +0,1631 |
| Aspect ratings only | +0,1808 | 5/5 | +0,1640 |

**Kết luận về lexicon mới.** Bản vá đảo chiều đóng góp của lexicon: trước đây `Text + lexicon` thấp hơn text-only (0,5556 so với 0,5579), nay cao hơn (0,5658). Mức tăng đúng là dương nhưng **chỉ +0,0079 Macro F1 và thua ở 2/5 fold**, nằm trong dao động giữa các fold, nên chưa đủ bằng chứng để một mình lexicon quyết định kiến trúc mô hình. Đề xuất với nhóm: giữ lexicon trong bộ đặc trưng vì nó không còn gây hại và tính được hoàn toàn từ text, nhưng nếu muốn khẳng định mức tăng này thì cần lặp CV nhiều lần với seed khác nhau. Riêng `sentiment_ratio` sau bản vá đã phân tách lớp rất rõ (Negative 0,090 — Neutral 0,428 — Positive 0,679), trước đó gần như phẳng.

**Kết luận về điểm khía cạnh.** Ghép năm điểm khía cạnh cải thiện dứt khoát đúng hai điểm yếu nhóm nêu, thắng 5/5 fold:

- **Neutral F1: 0,4456 → 0,6448** (+0,1992).
- **Recall Negative: 0,4563 → 0,6995** (+0,2432).
- Macro F1: 0,5579 → 0,7389 (+0,1809).

Cấu hình bàn giao chọn **text + lexicon + aspect** vì đạt cao nhất ở Macro F1, Neutral F1, Negative F1 và Accuracy. Đáng chú ý, `Aspect ratings only` có Recall Negative cao nhất (0,7676) nhưng **Neutral F1 thấp hơn** (0,6353 so với 0,6448): phần văn bản vẫn đóng góp riêng cho việc tách lớp Neutral, nên hybrid không phải chỉ đọc lại điểm số.

Năm cột khía cạnh không có giá trị thiếu (0,00% trên cả năm cột), nên bước `fillna(0.0)` trong `FeatureExtractor._prepare_numeric` không kích hoạt và không tạo giá trị 0 giả nằm ngoài thang 1-5. Hai cột `pos_e`, `neg_e` vẫn là hằng số 0 nên thực chất chỉ có 8 trong 10 cột số mang thông tin.

**Giới hạn phải ghi khi trình bày bảng này.** Nhãn `sentiment` được suy ra từ `Rating`, và điểm khía cạnh tương quan Spearman 0,5423-0,7368 với `Rating`. Vì vậy phần lớn mức tăng của nhóm aspect đến từ việc mô hình khôi phục lại thang điểm đã sinh ra nhãn, chứ không phải từ việc hiểu văn bản tốt hơn. Ngoài ra artifact hybrid cần đủ năm điểm khía cạnh lúc dự đoán, trong khi Web Demo chỉ nhận văn bản tự do. Kết luận vận hành: **hybrid dùng cho phần thực nghiệm và bảng so sánh mô hình; Web Demo giữ artifact text-only.** Hai artifact dùng chung split và seed nên số liệu đặt cạnh nhau được. Với riêng Web Demo, hướng cải thiện hợp lệ là `Text + lexicon` vì lexicon tính được từ chính văn bản người dùng nhập.

Ma trận text-only cuối cùng:

- `X_train` (development): 6.730 × 5.000.
- `X_test` (final test khóa): 1.683 × 5.000.
- Nhãn development: 4.964 Positive, 1.310 Neutral, 456 Negative.
- TV2 không báo cáo phân phối chi tiết hoặc metric mô hình trên final test ngoài việc kiểm tra contract kỹ thuật.

### Xử lý mất cân bằng

Hai chiến lược được bàn giao để TV3 đánh giá chéo:

1. Dùng `class_weight='balanced'` trong các mô hình hỗ trợ trọng số lớp.
2. Áp dụng SMOTE **bên trong từng fold development**. Minh họa tạo 4.964 mẫu cho mỗi lớp, tổng cộng 14.892 mẫu; final test không được resample.

Do dữ liệu TF-IDF có số chiều cao, SMOTE có thể tạo các vector tổng hợp khó diễn giải. Kết luận chọn chiến lược phải dựa trên Macro F1, Recall lớp Negative và ma trận nhầm lẫn, không chỉ dựa trên Accuracy.

## Tệp bàn giao

- `notebooks/01_data_exploration_eda.ipynb`: notebook EDA và feature engineering đã chạy từ đầu đến cuối.
- `src/features.py`: pipeline text/structured rõ ràng, xử lý text trùng, chia phân tầng, SMOTE và artifact contract.
- `models/text_tfidf_vectorizer.joblib`: vectorizer text-only.
- `models/text_feature_extractor.joblib`: extractor text-only đã fit trên development.
- `models/train_test_features.joblib`: ma trận development/final-test, nhãn, indices, feature names và metadata.
- `models/artifact_manifest.json`: runtime versions, dataset hash, Git SHA và checksum artifact.
- `requirements.lock`: môi trường Python 3.11 tái lập được để đọc artifact.
- `scripts/run_aspect_hybrid_ablation.py`: ablation text/lexicon/aspect kèm chỉ số từng lớp và so sánh theo cặp fold; xuất `reports/aspect_hybrid_ablation.csv` và `reports/aspect_hybrid_ablation_per_fold.csv`.
- `scripts/build_hybrid_artifacts.py`: dựng artifact hybrid, không đụng tới artifact text-only.
- `models/hybrid_feature_extractor.joblib`: extractor TF-IDF + 5 đặc trưng lexicon + 5 điểm khía cạnh, fit trên development (dùng `transform_hybrid`).
- `models/hybrid_train_test_features.joblib`: ma trận 6.730 × 5.010 và 1.683 × 5.010, cùng split/seed với bản text-only.
- `models/hybrid_artifact_manifest.json`: feature contract của bản hybrid, kèm ràng buộc dữ liệu lúc inference.
- `data/annotation/sentiment_audit_blind.csv` và `sentiment_audit_key.csv`: bộ 300 review cho hai người gán nhãn thủ công độc lập.
- `reports/overview_for_team.md`: giải thích pipeline bằng ngôn ngữ đơn giản.
- `reports/figures/`: chín biểu đồ EDA/feature diagnostics độ phân giải 300 dpi.
