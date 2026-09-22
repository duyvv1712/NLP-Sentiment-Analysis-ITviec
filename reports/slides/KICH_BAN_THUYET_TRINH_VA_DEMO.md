# Kịch bản thuyết trình và demo đồ án NLP — 2 người

## 1. Phân công

- **Người 1 — Thuyết trình:** trình bày liên tục toàn bộ slide 1–15, sau đó bàn giao một lần cho người demo.
- **Người 2 — Phạm Thành Trung:** phụ trách toàn bộ phần demo ứng dụng Streamlit và kết thúc buổi trình bày.
- **Thời lượng mục tiêu:** khoảng **18 phút 45 giây**; có thể rút xuống 15 phút theo phương án ở cuối tài liệu.

Nguồn nội dung: `reports/slides/NLP_ITviec_Sentiment_Slides.pptx` và ứng dụng hiện tại tại `http://localhost:8502`.

## 2. Phân bổ thời gian

| Phần | Người phụ trách | Thời lượng |
|---|---|---:|
| Slide 1–3: Bài toán và kiến trúc | Người 1 | 2 phút 15 giây |
| Slide 4–6: Tiền xử lý, EDA và đặc trưng | Người 1 | 3 phút |
| Slide 7–10: Mô hình và đánh giá | Người 1 | 4 phút 35 giây |
| Slide 11–12: Insight và giới thiệu ứng dụng | Người 1 | 1 phút 20 giây |
| Slide 13–15: Minh bạch, hạn chế và kết luận | Người 1 | 2 phút 05 giây |
| Demo trực tiếp và kết thúc | Phạm Thành Trung | 5 phút 30 giây |
| **Tổng** | **Cả hai** | **Khoảng 18 phút 45 giây** |

## 3. Nội dung cần đồng bộ trên slide trước khi trình bày (Đã hoàn tất đồng bộ)

Các mục dưới đây đã được sửa trực tiếp trong `scripts/build_presentation_slides.py` và biên dịch lại vào file `reports/slides/NLP_ITviec_Sentiment_Slides.pptx`, bảo đảm nội dung trên màn hình khớp hoàn toàn với kịch bản và ứng dụng thực tế:

1. [x] **Slide 1, 3 và 14:** Đã cập nhật thành **55/55 test passed** (55 bài kiểm thử) khớp với kết quả kiểm thử thực tế.
2. [x] **Slide 3:** Đã cập nhật thành **Segmented control** chọn 2 mô hình (thay cho radio).
3. [x] **Slide 12:** Đã đồng bộ cấu trúc 4 trang chính xác của ứng dụng Streamlit:
   - Tổng quan
   - Insight doanh nghiệp
   - Phân tích review
   - Mô hình & đánh giá
   (Đồng thời mockup giao diện đã đổi sang dạng Segmented control).
4. [x] **Slide 13:** Đã điều chỉnh câu chữ, xác định rõ là **trọng số TF-IDF nổi bật trong đầu vào** và ghi chú minh bạch là hệ thống chưa triển khai SHAP/LIME để tránh ngộ nhận về mặt XAI nhân quả.
5. [x] **Slide 6:** Đã cập nhật số liệu chuẩn: Text + Lexicon + Aspect đạt **Macro F1 0,7433** (thay cho 0,7369 cũ).
6. [x] **Slide 5:** Đã điều chỉnh layout và cỡ chữ chú thích dưới 3 biểu đồ EDA, bảo đảm không bị tràn mép hay cắt lẹm dòng thứ hai.
7. [x] **Slide 10:** Đã đổi cụm từ thành **“phát hiện 3 cụm phủ định”** để phản ánh đúng bản chất kỹ thuật Negation Scope & Lexicon.
8. [x] **Slide 15:** Khung bên phải đã đổi thành **“TIẾP THEO BƯỚC VÀO: DEMO TRỰC TIẾP — Phạm Thành Trung trình bày”**, tạo điểm chuyển tiếp mượt mà sang phần demo ứng dụng trước khi hội đồng hỏi đáp.

---

# PHẦN A — NGƯỜI 1 THUYẾT TRÌNH SLIDE

## Slide 1 — Giới thiệu đề tài

**Thời gian:** 35 giây

> Em xin chào thầy và các bạn. Nhóm em xin trình bày đồ án phân tích cảm xúc đánh giá nhân viên ngành Công nghệ Thông tin trên ITviec.
>
> Mục tiêu của nhóm không chỉ là phân loại review thành Tích cực, Trung tính và Tiêu cực, mà còn xây dựng một pipeline NLP hoàn chỉnh, so sánh nhiều mô hình và đưa kết quả vào một ứng dụng có thể sử dụng trực tiếp. Dữ liệu của nhóm gồm 8.417 review và hệ thống hiện đã vượt qua 55 bài kiểm thử.

## Slide 2 — Bối cảnh và thách thức

**Thời gian:** 55 giây

> Review trên ITviec phản ánh khá trực tiếp trải nghiệm về môi trường, quản lý, lương thưởng và cơ hội phát triển tại các công ty công nghệ. Nếu đọc thủ công hàng nghìn review thì rất tốn thời gian, vì vậy nhóm đặt mục tiêu tự động phân loại cảm xúc và khai thác insight doanh nghiệp.
>
> Bài toán có bốn thách thức chính. Thứ nhất là dữ liệu mất cân bằng khoảng 11 trên 1 giữa lớp Tích cực và Tiêu cực. Thứ hai là review chứa nhiều teencode và thuật ngữ IT Anh–Việt. Thứ ba là một review có thể vừa khen vừa chê. Cuối cùng là các cấu trúc phủ định như “không thân thiện” hoặc “ít cơ hội học hỏi”, rất dễ làm mô hình hiểu sai sắc thái.

## Slide 3 — Kiến trúc tổng thể

**Thời gian:** 45 giây

> Hệ thống gồm sáu bước. Review được thu thập và làm sạch, sau đó chuyển thành vector TF-IDF; ở cấu hình mở rộng, vector này được ghép thêm năm đặc trưng Lexicon. Đầu vào được đưa vào các mô hình học máy và Stacking Ensemble. Kết quả xác suất tiếp tục đi qua bước hiệu chỉnh quyết định trước khi trả về nhãn cuối trên ứng dụng Streamlit.
>
> Nhóm tách riêng dữ liệu huấn luyện, cross-validation và tập đánh giá; đồng thời sử dụng hai pipeline tiền xử lý khác nhau cho Machine Learning truyền thống và Transformer.

## Slide 4 — Tiền xử lý và Negation Scope

**Thời gian:** 1 phút 10 giây

> Với nhóm mô hình truyền thống, văn bản được chuẩn hóa Unicode, chuyển emoji thành tín hiệu sắc thái, chuẩn hóa teencode, tách từ tiếng Việt và loại stopword. Tuy nhiên, các từ phủ định như “không”, “chưa” và “ít” được giữ lại vì chúng quyết định chiều cảm xúc.
>
> Với ViSoBERT, nhóm xử lý nhẹ hơn: giữ dấu câu, cấu trúc câu và ngữ cảnh để tokenizer BPE của Transformer tự xử lý.
>
> Điểm đáng chú ý là Negation Scope Detection. Trong câu “môi trường làm việc không được thân thiện”, nếu chỉ nhìn từ “thân thiện”, mô hình có thể nghiêng về Tích cực. Hệ thống phát hiện phạm vi phủ định và chuyển tín hiệu của cụm này sang Tiêu cực.

## Slide 5 — Khám phá dữ liệu

**Thời gian:** 40 giây

> Qua EDA, nhóm ghi nhận 73,8% review Tích cực, 19,5% Trung tính và chỉ 6,8% Tiêu cực. Vì vậy, Accuracy cao chưa chắc mô hình hoạt động tốt trên cả ba lớp.
>
> Độ dài review phân tán khá lớn, trong khi các khía cạnh như quản lý, đào tạo và lương thưởng có tương quan đáng kể với rating. Đây là cơ sở để nhóm ưu tiên Macro F1 và Recall lớp Tiêu cực khi đánh giá.

## Slide 6 — So sánh nhóm đặc trưng

**Thời gian:** 1 phút 10 giây

> Nhóm thực hiện ablation để kiểm tra việc thêm Lexicon có thực sự giúp mô hình hay không.
>
> Khi bổ sung năm đặc trưng Lexicon vào 5.000 chiều TF-IDF, Macro F1 cross-validation tăng từ khoảng 0,558 lên 0,566; Recall Tiêu cực tăng từ khoảng 45,6% lên 48%. Mức tăng không quá lớn nhưng ổn định và tập trung vào lớp khó nhất.
>
> Các aspect rating cho kết quả rất cao, trên 0,73 Macro F1, nhưng nhóm chủ động loại bỏ vì chúng gần trùng với cách tạo nhãn. Nếu giữ lại, mô hình có thể học đường tắt từ rating thay vì thực sự hiểu nội dung văn bản.

## Slide 7 — Các mô hình và Stacking Ensemble

**Thời gian:** 1 phút 15 giây

> Nhóm so sánh Logistic Regression, Linear SVM, Random Forest và Multinomial Naive Bayes. Mỗi mô hình có thế mạnh khác nhau: Naive Bayes nhạy với tần suất từ, Logistic Regression và SVM phù hợp với không gian TF-IDF nhiều chiều, còn Random Forest mô hình hóa quan hệ phi tuyến.
>
> Stacking sử dụng ba mô hình nền là Naive Bayes, Logistic Regression và Linear SVM. Dự đoán của chúng được đưa vào Logistic Regression tầng trên để học cách kết hợp.
>
> Kết quả Stacking đạt CV Macro F1 0,5619 trên tập phát triển và Accuracy khoảng 77,66% trên Final Test đã khóa.

## Slide 8 — ViSoBERT

**Thời gian:** 55 giây

> Nhóm cũng đánh giá ViSoBERT ở chế độ zero-shot. Mô hình đạt Macro F1 khoảng 0,4036 và Accuracy 65,36%. Điểm đáng chú ý là Recall Tiêu cực cao, khoảng 75%, nhưng Recall Trung tính chỉ gần 5%.
>
> Nguyên nhân chính là mô hình chưa được fine-tune trên miền review ITviec và pipeline tiền xử lý cổ điển có thể làm mất thông tin mà Self-Attention cần. Kết quả này cho thấy Transformer không tự động tốt hơn nếu chưa được thích nghi đúng dữ liệu và đúng preprocessing.

## Slide 9 — Kết quả và ma trận nhầm lẫn

**Thời gian:** 1 phút 10 giây

> Đây là kết quả trên 1.683 review đánh giá. Stacking có Accuracy 77,66% nhưng Macro F1 chỉ khoảng 0,5475. Sự chênh lệch này chính là bẫy Accuracy trong dữ liệu mất cân bằng.
>
> Nhìn vào ma trận nhầm lẫn, mô hình nhận đúng 1.163 review Tích cực nhưng chỉ nhận đúng 30 trên 114 review Tiêu cực. Nhiều review Tiêu cực bị nhầm thành Trung tính hoặc Tích cực. Vì vậy, nếu chỉ báo cáo Accuracy thì chúng ta sẽ bỏ qua điểm yếu quan trọng nhất của hệ thống.

## Slide 10 — Error Analysis và ngưỡng 30%

**Thời gian:** 1 phút 15 giây

> Từ ma trận nhầm lẫn, nhóm đi sâu vào từng trường hợp sai và nhận thấy lỗi thường xuất hiện ở câu phủ định, câu châm biếm và review vừa khen vừa chê.
>
> Nhóm thử hạ ngưỡng nhận diện Tiêu cực xuống 30%. Recall Tiêu cực tăng từ 26,3% lên 35,1% và Macro F1 tăng khoảng 0,0032. Tuy nhiên, Precision Tiêu cực giảm từ khoảng 53,6% xuống 44,4%.
>
> Nghĩa là hệ thống bắt được thêm review Tiêu cực nhưng cũng tạo thêm cảnh báo nhầm. Đây là đánh đổi có chủ đích, không phải cải thiện miễn phí và cũng không cần huấn luyện lại mô hình.

## Slide 11 — Insight doanh nghiệp

**Thời gian:** 55 giây

> Ngoài dự đoán từng review, hệ thống còn tổng hợp insight theo doanh nghiệp. WordCloud giúp quan sát nhanh những từ thường xuất hiện trong nhóm review Tích cực hoặc Tiêu cực.
>
> Các chủ đề được khen thường liên quan đến môi trường trẻ, đồng nghiệp và cơ hội học hỏi. Những phản hồi tiêu cực thường đề cập đến OT, tốc độ tăng lương và quản lý.
>
> WordCloud chỉ phản ánh tần suất từ, không tự động chứng minh nguyên nhân. Vì vậy, ứng dụng cho phép mở review chi tiết để kiểm tra lại ngữ cảnh.

## Slide 12 — Giới thiệu ứng dụng

**Thời gian:** 25 giây

> Từ pipeline nghiên cứu, nhóm đã xây dựng ứng dụng Streamlit với bốn khu vực: Tổng quan, Insight doanh nghiệp, Phân tích review và Mô hình & đánh giá. Trang cuối gom phần so sánh, chất lượng và phân tích lỗi vào một luồng thống nhất.
>
> Sau khi trình bày phần kết quả và hạn chế, nhóm sẽ demo trực tiếp các chức năng này trên hệ thống hiện tại.

## Slide 13 — Minh bạch mô hình

**Thời gian:** 35 giây

> Giao diện cung cấp các token và đặc trưng TF-IDF nổi bật để người dùng biết văn bản nào thực sự đi vào mô hình.
>
> Cần lưu ý đây là trọng số đầu vào TF-IDF, không phải mức đóng góp nhân quả như SHAP hoặc LIME. Phần giải thích đáng tin cậy nhất hiện tại là việc tách riêng dự đoán Stacking, bước hiệu chỉnh ngưỡng và tín hiệu phủ định của Lexicon.

## Slide 14 — Thành tựu và hạn chế

**Thời gian:** 50 giây

> Nhóm đã hoàn thiện pipeline tiền xử lý tiếng Việt, so sánh nhiều nhóm đặc trưng, xây dựng Stacking Ensemble, đánh giá ViSoBERT và triển khai ứng dụng Streamlit. Hệ thống hiện vượt qua 55 bài kiểm thử.
>
> Tuy nhiên, Recall lớp Tiêu cực vẫn còn hạn chế do dữ liệu mất cân bằng. ViSoBERT mới dừng ở zero-shot và hệ thống chưa phân tích cảm xúc chi tiết theo từng khía cạnh. Nhóm xem đây là giới hạn cần công khai, không che giấu bằng Accuracy tổng thể.

## Slide 15 — Kết luận và bàn giao cho phần demo

**Thời gian:** 40 giây

> Tóm lại, kết quả quan trọng nhất của đồ án là một pipeline NLP hoàn chỉnh, từ dữ liệu, tiền xử lý, mô hình, đánh giá cho đến ứng dụng thực tế.
>
> Trong tương lai, nhóm có thể phát triển phân tích cảm xúc theo khía cạnh, fine-tune ViSoBERT trên review IT, mở rộng dữ liệu và ứng dụng LLM để tự động tổng hợp insight doanh nghiệp.
>
> Phần trình bày slide của em xin kết thúc tại đây. Tiếp theo, em xin mời bạn Phạm Thành Trung demo trực tiếp hệ thống của nhóm.

**Thao tác:** Người 1 kết thúc toàn bộ slide tại slide 15. Trung chuyển sang cửa sổ trình duyệt đã mở sẵn. Sau thời điểm này không quay lại PowerPoint.

---

# PHẦN B — PHẠM THÀNH TRUNG DEMO WEB

## Chuẩn bị trước buổi trình bày

- Khởi động app và kiểm tra `http://localhost:8502/_stcore/health` trả về `ok`.
- Mở sẵn `http://localhost:8502` và chờ các biểu đồ tải hoàn tất.
- Để trình duyệt ở mức zoom 100%, ẩn bookmark bar và tắt thông báo Windows.
- Chạy thử các mẫu có sẵn một lần để model và bộ tiền xử lý nằm trong cache.
- Không cập nhật package hoặc pull code ngay trước giờ trình bày.

## Demo 1 — Mở đầu và Insight doanh nghiệp

**Thời gian:** 1 phút

**Thao tác:** Mở **Insight doanh nghiệp** và chỉ nhanh vào phần tổng hợp cảm xúc, xu hướng và từ khóa.

> Em xin chào thầy và các bạn. Em là Phạm Thành Trung, đại diện nhóm trình bày phần demo.
>
> Đầu tiên, em vào trang Insight doanh nghiệp. Trang này tổng hợp cảm xúc, xu hướng và từ khóa để người dùng nhanh chóng nắm được bức tranh chung của doanh nghiệp.

**Thao tác:** Chuyển WordCloud giữa **Tích cực** và **Tiêu cực**.

> Đây là bản đồ từ khóa, cho thấy những từ xuất hiện thường xuyên trong review. Người dùng có thể chuyển giữa nhóm Tích cực và Tiêu cực để xem từ khóa đặc trưng của từng nhóm. Từ càng lớn nghĩa là xuất hiện càng nhiều, chứ chưa thể kết luận đó là nguyên nhân trực tiếp tạo ra cảm xúc.

**Thao tác:** Kéo xuống **Góc nhìn review** và mở một review.

> Phần Góc nhìn review giúp người dùng đối chiếu các thống kê bên trên với nội dung và ngữ cảnh thực tế của từng đánh giá.

## Demo 2 — Mô hình & đánh giá

**Thời gian:** 1 phút 30 giây

### So sánh mô hình

**Thao tác:** Mở **Mô hình & đánh giá** và giữ mục **So sánh mô hình**.

> Ở mục So sánh mô hình, các mô hình được xếp hạng bằng 5-fold cross-validation trên tập train. Stacking đạt CV Macro F1 cao nhất, khoảng 0,5619, nên được chọn để đánh giá một lần trên Final Test.
>
> Bảng bên dưới cho thấy trên Final Test, Stacking có Macro F1 cao hơn và nhận diện ba lớp cân bằng hơn ViSoBERT zero-shot, đặc biệt ở lớp Trung tính.

### Chất lượng mô hình

**Thao tác:** Chuyển sang **Chất lượng mô hình**, giữ ngưỡng **30%** và chỉ vào ma trận nhầm lẫn.

> Accuracy cho biết tỷ lệ dự đoán đúng trên toàn bộ review, còn Recall Tiêu cực cho biết hệ thống tìm đúng bao nhiêu review Tiêu cực.
>
> Với ngưỡng 30%, hệ thống nhận đúng 40 trong 114 review Tiêu cực, tương đương Recall 35,1%. Trong ma trận, 40 review được nhận đúng là Tiêu cực, 40 bị nhầm sang Trung tính và 34 bị nhầm sang Tích cực. Vì vậy, dù kết quả chung khá tốt, hệ thống vẫn còn bỏ sót nhiều review Tiêu cực.

### Lỗi và ngưỡng

**Thao tác:** Chuyển sang **Lỗi & ngưỡng**, kéo ngưỡng từ **30%** xuống **10%**, sau đó mở một review dự đoán sai.

> Ở ngưỡng 30%, Recall Tiêu cực là 35,1% và Precision là 44,4%. Khi hạ xuống 10%, Recall tăng lên 77,2% nhưng Precision giảm còn 25,2%. Nghĩa là hệ thống bắt được nhiều review Tiêu cực hơn, đổi lại cảnh báo nhầm cũng nhiều hơn.
>
> Review lỗi này được gắn nhãn Tiêu cực theo rating nhưng model dự đoán Trung tính. Nội dung vừa khen môi trường và công nghệ, vừa chê OT nhiều và tăng lương chậm. Vì có cả khen lẫn chê nên model dễ nhầm; đồng thời rating cũng chưa nói hết sắc thái của nội dung review.

## Demo 3 — Phân tích review bằng mẫu thử

**Thời gian:** 3 phút

### Giới thiệu chức năng

**Thao tác:** Mở **Phân tích review** và giữ **Text-only · 5.000 chiều**.

> Đây là nơi người dùng nhập một review mới để hệ thống dự đoán cảm xúc. Text-only sử dụng đặc trưng từ nội dung; còn Text + Lexicon ghép thêm năm tín hiệu từ điển cảm xúc vào vector model.
>
> Trong phần demo, em giữ Text-only để thống nhất với phần đánh giá. Các tín hiệu Lexicon và phạm vi phủ định vẫn được kiểm tra tại Decision Gate trước khi trả nhãn cuối.

### Mẫu Lời khen

**Thao tác:** Chọn mẫu **Lời khen** và bấm **Phân tích cảm xúc**.

> Đầu tiên, em thử mẫu Lời khen. Hệ thống nhận diện review này là Tích cực. Ba thanh bên dưới cho biết xác suất model dự đoán cho từng lớp; thanh Tích cực cao nhất, khoảng 77,5%, nên được chọn làm kết quả.

**Thao tác:** Kéo xuống phần pipeline, token và TF-IDF.

> Với chính lời khen này, hệ thống nhận nội dung, làm sạch và tách từ tiếng Việt. TF-IDF chuyển các từ thành số để ba model trong Stacking đưa ra dự đoán. Sau đó, Decision Gate kiểm tra thêm ngưỡng và tín hiệu ngôn ngữ rồi trả về nhãn Tích cực cùng các thanh xác suất vừa thấy.
>
> Đây là review sau khi làm sạch và tách từ. Những tiếng thuộc cùng một từ được nối bằng dấu gạch dưới, ví dụ như “môi_trường”. Văn bản này được transform bằng bộ TF-IDF đã fit trên tập train; hệ thống không train lại trên review vừa nhập.
>
> Biểu đồ cho thấy những từ nổi bật trong review theo TF-IDF. Cột cao hơn nghĩa là từ có trọng số đầu vào lớn hơn, nhưng không có nghĩa chỉ riêng từ đó quyết định kết quả.

### Mẫu Ý kiến hỗn hợp

**Thao tác:** Chọn **Ý kiến hỗn hợp** và phân tích.

> Tiếp theo, em thử mẫu Ý kiến hỗn hợp, có cả điểm ổn và điểm chưa tốt. Review vừa khen “công việc ổn”, vừa chê “quy trình chậm”, nên model nghiêng về Trung tính khoảng 60,7%. Nhóm Tích cực vẫn có khoảng 29,1% vì nội dung không hoàn toàn là lời chê.

### Mẫu Lời phàn nàn

**Thao tác:** Chọn **Lời phàn nàn** và phân tích.

> Mẫu này có sắc thái rõ ràng hơn, với nhiều lời chê như “rất tệ”, “quản lý yếu” và “lương thấp”. Các ý đều nghiêng về một hướng nên hệ thống nhận diện là Tiêu cực, khoảng 91,1%.

### Mẫu Cần lưu ý và phạm vi phủ định

**Thao tác:** Chọn **Cần lưu ý**, phân tích rồi mở phần **Quy tắc & giới hạn**.

> Cuối cùng, em thử mẫu Cần lưu ý. Ba nhóm có xác suất khá gần nhau: Tiêu cực 35,4%, Trung tính 34,4% và Tích cực 30,2%. Hệ thống chọn Tiêu cực vì nhỉnh hơn hai nhóm còn lại; ngưỡng 30% không làm đổi nhãn trong ví dụ này. Vì vậy, người dùng vẫn cần đọc nội dung review để hiểu kết quả.
>
> Stacking đưa ra dự đoán trước, sau đó Decision Gate kiểm tra thêm ngưỡng và cách dùng từ. Ví dụ, “minh bạch” là ý tốt nhưng “thiếu minh bạch” lại là lời chê; “lương tương xứng” cũng khác với “lương chưa tương xứng”. Hệ thống xét cả cụm từ như vậy để xử lý phạm vi phủ định. Trong review này, Stacking đã nghiêng về Tiêu cực nên nhãn cuối không thay đổi.

## Kết thúc demo và buổi trình bày

> Vừa rồi là toàn bộ phần demo, từ bức tranh cảm xúc của doanh nghiệp đến cách hệ thống phân tích và nhận diện cảm xúc cho từng review mới.
>
> Phần demo của em đến đây là kết thúc. Nhóm 4 xin chân thành cảm ơn thầy và các bạn đã theo dõi và xin sẵn sàng trả lời câu hỏi.

**Thao tác:** Giữ nguyên trang Phân tích review. Không chuyển lại PowerPoint.

---

# 4. Phương án rút xuống đúng 15 phút

Nếu thời gian bị giới hạn nghiêm ngặt, thực hiện các cắt giảm sau:

1. Slide 5 chỉ nói phân bố nhãn, bỏ phần độ dài và tương quan.
2. Slide 8 chỉ nói ViSoBERT chưa fine-tune và yếu hơn Stacking về tổng thể.
3. Demo Insight không mở review chi tiết.
4. Demo dự đoán chỉ chạy Text-only; nói ngắn rằng Text + Lexicon là cấu hình so sánh thứ hai.
5. Slide 13 trình bày trong 20 giây.

Tổng thời gian sau khi rút gọn: khoảng **15 phút**.

# 5. Xử lý sự cố khi demo

## App tải chậm

> Trong lúc hệ thống xử lý, em xin nhắc lại rằng model và bộ TF-IDF được load cục bộ; lần phân tích đầu tiên có thể lâu hơn do khởi tạo cache.

## WordCloud chưa hiện ngay

Chuyển sang phần cơ cấu cảm xúc trước, sau đó quay lại WordCloud khi ảnh đã tải xong.

## Kết quả khác con số đã tập trước

Không cố giải thích theo số cũ. Nói:

> Đây là kết quả trực tiếp của lượt suy luận hiện tại. Em sẽ tập trung giải thích luồng xử lý và lý do hệ thống chọn nhãn đang hiển thị.

## App không truy cập được

Trung không sửa lỗi trực tiếp trước hội đồng. Dựa trên nội dung ứng dụng mà Người 1 đã giới thiệu ở slide 12, Trung tóm tắt ngắn bốn khu vực chức năng, nêu kết quả dự đoán đã chuẩn bị và chuyển thẳng sang lời cảm ơn. Không quay lại PowerPoint.

# 6. Ba câu hỏi hội đồng dễ hỏi

## Vì sao Accuracy cao nhưng Macro F1 thấp?

> Vì lớp Tích cực chiếm gần 74% dữ liệu. Mô hình dự đoán tốt lớp đa số sẽ có Accuracy cao, nhưng Macro F1 tính cân bằng ba lớp nên phản ánh rõ điểm yếu ở Trung tính và Tiêu cực.

## Vì sao không dùng aspect rating dù kết quả rất cao?

> Vì aspect rating gần trùng với cách tạo nhãn mục tiêu, gây nguy cơ Data Shortcut. Nhóm loại bỏ để đánh giá đúng khả năng hiểu văn bản của mô hình.

## Vì sao chọn ngưỡng Negative 30%?

> Đây là một phương án hậu xử lý nhằm tăng khả năng phát hiện lớp Tiêu cực mà không huấn luyện lại. Kết quả cho thấy Recall tăng nhưng Precision giảm, nên nhóm trình bày rõ đây là một đánh đổi chứ chưa khẳng định 30% là ngưỡng tối ưu tuyệt đối.
