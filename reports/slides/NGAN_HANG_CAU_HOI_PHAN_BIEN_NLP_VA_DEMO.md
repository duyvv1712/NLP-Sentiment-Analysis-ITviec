# Câu hỏi phản biện trọng tâm — Đồ án NLP ITviec

Giảng viên chỉ hỏi khoảng ba câu, nhưng không thể biết trước câu nào. Bộ này giữ **18 câu NLP cơ bản và có xác suất cao**, chia thành sáu chủ đề. Các câu **⭐** cần ưu tiên học trước.

---

## 1. Bài toán và dữ liệu

### 1.1 ⭐ Bài toán NLP của nhóm là gì?

**Trả lời:** Nhóm phân loại nội dung review ITviec thành Positive, Neutral và Negative; sau đó tổng hợp insight cảm xúc theo doanh nghiệp và đưa pipeline vào Web Demo.

### 1.2 Nhãn cảm xúc được tạo như thế nào?

**Trả lời:** Nhãn được suy ra từ rating: 4–5 sao là Positive, 3 sao là Neutral và 1–2 sao là Negative. Đây là weak label nên có thể tồn tại trường hợp điểm sao không hoàn toàn khớp nội dung.

### 1.3 ⭐ Dữ liệu có đặc điểm gì đáng chú ý?

**Trả lời:** Tập dữ liệu có 8.417 review; Positive chiếm khoảng 73,8%, Neutral 19,5% và Negative chỉ 6,8%. Mất cân bằng lớp là nguyên nhân Accuracy không đủ để đánh giá model.

---

## 2. Tiền xử lý tiếng Việt

### 2.1 ⭐ Pipeline tiền xử lý gồm những bước nào?

**Trả lời:** Chuẩn hóa Unicode, loại URL/email và nhiễu, chuẩn hóa teencode, xử lý emoji, tách từ tiếng Việt, bảo toàn cụm từ, xử lý phủ định và loại stopword có kiểm soát.

### 2.2 Vì sao tiếng Việt cần tách từ và chuẩn hóa teencode?

**Trả lời:** Một từ tiếng Việt có thể gồm nhiều tiếng như “môi trường”. Teencode như “ko”, “k”, “cty” còn làm cùng một ý bị chia thành nhiều token; chuẩn hóa giúp giảm độ thưa của từ vựng.

### 2.3 ⭐ Vì sao không loại hết stopword?

**Trả lời:** Nhóm phải giữ “không”, “chưa”, “ít”, “thiếu” vì chúng có thể đảo cực tính. Nếu xóa “không” khỏi “không tốt”, ý nghĩa câu sẽ bị thay đổi hoàn toàn.

---

## 3. Phủ định và biểu diễn đặc trưng

### 3.1 ⭐ Negation Scope Detection là gì?

**Trả lời:** Đây là bước xác định từ phủ định tác động lên cụm cảm xúc nào. Ví dụ “thân thiện” là tích cực, nhưng “không được thân thiện” phải trở thành tín hiệu tiêu cực.

### 3.2 TF-IDF là gì và vì sao nhóm sử dụng?

**Trả lời:** TF-IDF tăng trọng số cho từ quan trọng trong một review nhưng không quá phổ biến trong toàn corpus. Nó nhanh, dễ kiểm tra và phù hợp các mô hình tuyến tính trên dữ liệu văn bản sparse.

### 3.3 Lexicon là gì và được kết hợp thế nào?

**Trả lời:** Lexicon là từ điển từ/cụm từ cảm xúc. Nhóm nối năm đặc trưng Lexicon gồm từ và emoji tích cực/tiêu cực cùng tỷ lệ cảm xúc vào 5.000 chiều TF-IDF, tạo vector 5.005 chiều.

---

## 4. Mô hình NLP

### 4.1 ⭐ Stacking hoạt động như thế nào?

**Trả lời:** Naive Bayes, Logistic Regression và Linear SVM tạo dự đoán cấp một; Logistic Regression tầng meta học cách kết hợp chúng. Dữ liệu cho tầng meta cần được tạo theo out-of-fold để tránh leakage.

### 4.2 Vì sao các mô hình tuyến tính phù hợp TF-IDF?

**Trả lời:** TF-IDF tạo ma trận sparse nhiều chiều. Logistic Regression và Linear SVM xử lý kiểu dữ liệu này hiệu quả và thường là baseline mạnh cho phân loại văn bản.

### 4.3 ⭐ Vì sao không chọn ViSoBERT làm model cuối?

**Trả lời:** ViSoBERT mới được chạy zero-shot, chưa fine-tune trên miền ITviec. Macro F1 khoảng 0,4036, thấp hơn pipeline ML; Negative Recall cao nhưng Neutral Recall chỉ khoảng 4,88%.

---

## 5. Đánh giá mô hình

### 5.1 ⭐ Vì sao ưu tiên Macro F1 thay vì chỉ nhìn Accuracy?

**Trả lời:** Positive chiếm gần 74%, nên model có thể đạt Accuracy cao dù bỏ sót Negative. Macro F1 cho ba lớp trọng số ngang nhau, phản ánh cân bằng hơn.

### 5.2 Confusion Matrix và Error Analysis cho biết gì?

**Trả lời:** Confusion Matrix cho biết model nhầm lớp nào với lớp nào; Error Analysis đọc từng review sai để tìm nguyên nhân. Baseline chỉ nhận đúng 30/114 review Negative, cho thấy bỏ sót Negative là lỗi chính.

### 5.3 ⭐ Vì sao hạ ngưỡng Negative xuống 0.30?

**Trả lời:** Để giảm bỏ sót Negative mà không cần train lại. Recall tăng từ 26,32% lên 35,09%, nhưng Precision giảm từ 53,57% xuống 44,44%; đây là trade-off và chưa đạt kỳ vọng Recall 55–60%.

---

## 6. Pipeline NLP trên Web Demo

### 6.1 ⭐ Một review mới được xử lý theo luồng nào?

**Trả lời:** Kiểm tra input → tiền xử lý → transform bằng extractor đã fit → thêm Lexicon nếu được chọn → Stacking dự đoán → threshold/Hybrid Gate → hiển thị nhãn và trace.

### 6.2 Vì sao phải load cả model và TF-IDF vectorizer?

**Trả lời:** Model không nhận trực tiếp văn bản thô. Review phải được transform bằng đúng extractor đã fit khi train; demo chỉ gọi `transform`, không fit lại vectorizer.

### 6.3 Tại sao xác suất model thô khác kết quả cuối?

**Trả lời:** Các thanh xác suất là đầu ra thô của Stacking. Nhãn và confidence cuối có thể đã qua ngưỡng Negative hoặc Hybrid Decision Gate, nên giao diện hiển thị trace để chỉ rõ bước hiệu chỉnh.

---

## Thứ tự ôn tập

1. Học chắc **10 câu có dấu ⭐**.
2. Đọc qua 8 câu còn lại để có phương án dự phòng.
3. Nhớ các số: **8.417 review**, Negative **6,8%**, TF-IDF **5.000 chiều**, Recall Negative **26,32% → 35,09%**.

> Nếu bị hỏi sâu về threshold: ngưỡng 0.30 hiện được khảo sát post-hoc trên final test đã đánh giá. Quy trình chặt hơn phải chọn threshold trên validation rồi mới đánh giá trên test độc lập.
