# Kịch bản thuyết trình và demo đồ án NLP — 2 người

## 1. Phân công

- **Người 1 — Thuyết trình:** trình bày liên tục toàn bộ slide 1–15, sau đó bàn giao một lần cho người demo.
- **Người 2 — Phạm Thành Trung:** phụ trách toàn bộ phần demo ứng dụng Streamlit và kết thúc buổi trình bày.
- **Thời lượng mục tiêu:** khoảng **17 phút 05 giây**; có thể rút xuống 15 phút theo phương án ở cuối tài liệu.

Nguồn nội dung: `reports/slides/NLP_ITviec_Sentiment_Slides.pptx` và ứng dụng hiện tại tại `http://localhost:8502`.

## 2. Phân bổ thời gian

| Phần | Người phụ trách | Thời lượng |
|---|---|---:|
| Slide 1–3: Bài toán và kiến trúc | Người 1 | 2 phút 15 giây |
| Slide 4–6: Tiền xử lý, EDA và đặc trưng | Người 1 | 3 phút |
| Slide 7–10: Mô hình và đánh giá | Người 1 | 4 phút 35 giây |
| Slide 11–12: Insight và giới thiệu ứng dụng | Người 1 | 1 phút 20 giây |
| Slide 13–15: Minh bạch, hạn chế và kết luận | Người 1 | 2 phút 05 giây |
| Demo trực tiếp và kết thúc | Phạm Thành Trung | 3 phút 50 giây |
| **Tổng** | **Cả hai** | **Khoảng 17 phút 05 giây** |

## 3. Nội dung cần đồng bộ trên slide trước khi trình bày

Các mục dưới đây là lỗi của **file PowerPoint**, không phải lời thoại. Nên sửa trước khi quay hoặc trình bày để nội dung nhìn thấy trên màn hình không mâu thuẫn với người nói.

1. Slide 1, 3 và 14 đang ghi **43/43 unit test**; project hiện tại đã đạt **55 tests passed**.
2. Slide 3 ghi điều khiển bằng radio; ứng dụng hiện dùng **segmented control**.
3. Slide 12 mô tả 4 phân hệ và ảnh UI cũ; ứng dụng hiện có 4 trang:
   - Tổng quan
   - Insight doanh nghiệp
   - Phân tích review
   - Mô hình & đánh giá
4. Slide 13 không nên khẳng định TF-IDF là mức đóng góp nhân quả. Nên nói đây là **các đặc trưng TF-IDF nổi bật trong đầu vào**; hệ thống chưa triển khai SHAP hoặc LIME.
5. Slide 6 đang dùng số cũ. Artifact hiện tại cho Text + Lexicon đạt **Macro F1 0,5664** và **Recall Negative 48,0%** trên cross-validation; Text + Lexicon + Aspect đạt **0,7433**. Khi nói có thể làm tròn lần lượt thành **0,566**, **khoảng 48%** và **0,743**.
6. Chú thích dưới ba biểu đồ ở slide 5 đang chạm mép và bị cắt chữ; cần nâng chú thích lên hoặc giảm cỡ chữ trước khi xuất bản trình chiếu.
7. Slide 10 nên đổi cụm “bóc tách 3 cụm từ XAI” thành **“phát hiện 3 cụm phủ định”** để không đánh đồng luật Lexicon với phương pháp giải thích mô hình như SHAP/LIME.
8. Slide 15 đang ghi “Phần hỏi đáp” trong khi kịch bản còn phần demo. Nên đổi thành **“Chuyển sang demo trực tiếp”**; phần hỏi đáp chỉ bắt đầu sau khi Trung kết thúc demo.

---

# PHẦN A — NGƯỜI 1 THUYẾT TRÌNH SLIDE

## Slide 1 — Giới thiệu đề tài

**Thời gian:** 35 giây

> Em xin chào thầy và các bạn. Nhóm 4 xin trình bày đồ án phân tích cảm xúc đánh giá nhân viên ngành Công nghệ Thông tin trên ITviec.
>
> Mục tiêu của nhóm không chỉ là phân loại review thành Tích cực, Trung tính và Tiêu cực, mà còn xây dựng một pipeline NLP hoàn chỉnh, so sánh nhiều mô hình và đưa kết quả vào một ứng dụng có thể sử dụng trực tiếp. Dữ liệu của nhóm gồm 8.417 review và hệ thống hiện đã vượt qua 55 bài kiểm thử.

**Lưu ý:** Không cần đọc lại toàn bộ tên và vai trò thành viên trên slide.

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
- Chuẩn bị sẵn câu review trong clipboard:

> Môi trường làm việc không được thân thiện, đồng nghiệp không hỗ trợ và ít cơ hội học hỏi.

- Chạy thử câu này một lần trước khi trình bày để model và bộ tiền xử lý đã nằm trong cache.
- Không cập nhật package hoặc pull code ngay trước giờ trình bày.

## Demo 1 — Trang Tổng quan

**Thời gian:** 20 giây

**Thao tác:** Mở trang **Tổng quan**.

> Em là Phạm Thành Trung. Bây giờ em xin demo nhanh hệ thống NLP của nhóm. Trang Tổng quan tóm tắt số review, số doanh nghiệp và tỷ lệ của ba loại cảm xúc. Từ đây, mình có thể xem insight của từng công ty hoặc phân tích ngay một review mới.

**Không nên:** đọc lần lượt mọi KPI hoặc dừng quá lâu ở biểu đồ đầu tiên.

## Demo 2 — Insight doanh nghiệp

**Thời gian:** 45 giây

**Thao tác:**

1. Chọn **Insight doanh nghiệp** ở sidebar.
2. Chọn doanh nghiệp **FPT Software**.
3. Chỉ vào cơ cấu cảm xúc và xu hướng theo thời gian.
4. Chuyển WordCloud từ **Tích cực** sang **Tiêu cực**.

**Lời nói:**

> Ở trang Insight doanh nghiệp, em chọn FPT Software và giữ mức tối thiểu 50 review để kết quả không bị ảnh hưởng bởi một mẫu quá nhỏ. Khi em chuyển sang nhóm Tiêu cực, WordCloud và bảng từ khóa được tính lại theo đúng nhóm review này. Từ càng lớn nghĩa là xuất hiện càng nhiều, chứ không có nghĩa đó là lý do trực tiếp tạo ra cảm xúc.

## Demo 3 — Phân tích cảm xúc review

**Thời gian:** 2 phút

### Lượt chính — Text-only

**Thao tác:**

1. Chọn **Phân tích review**.
2. Chọn pipeline **Text-only · 5.000 chiều**.
3. Dán câu review đã chuẩn bị.
4. Bấm **Phân tích cảm xúc**.

**Kết quả hiện tại đã kiểm chứng:**

- Nhãn cuối: **Tiêu cực — 71,1%**.
- Xác suất Stacking ban đầu: Neutral khoảng **39,4%**, Negative khoảng **36,1%**, Positive khoảng **24,5%**.
- Hybrid phát hiện ba cụm phủ định:
  - `không được thân thiện`
  - `không hỗ trợ`
  - `ít cơ hội học hỏi`

**Lời nói:**

> Kết quả cuối là Tiêu cực, khoảng 71,1%. Tuy nhiên, kết quả ban đầu còn khá phân vân giữa Trung tính và Tiêu cực. Sau khi kiểm tra thêm ba cụm phủ định, hệ thống mới đưa ra nhãn cuối. Các thanh xác suất là điểm ban đầu của model, còn nhãn phía trên là kết quả sau bước hiệu chỉnh.

**Thao tác:** Kéo xuống khu vực pipeline gồm 3 giai đoạn và 6 bước.

> Pipeline được chia thành ba giai đoạn: hiểu văn bản, biểu diễn và dự đoán, cuối cùng là ra quyết định. Sáu ô bên trong lần lượt hiển thị review gốc, kết quả chuẩn hóa, vector TF-IDF, dự đoán Stacking, bước hiệu chỉnh và nhãn cuối. Đây là số liệu thật của lượt chạy; ứng dụng dùng bộ TF-IDF đã học từ trước và không học lại trên câu vừa nhập.

**Không chạy Text + Lexicon lần hai:** chỉ nói ngắn rằng cấu hình 5.005 đặc trưng được trình bày trong mục So sánh mô hình.

## Demo 4 — Mô hình & đánh giá

**Thời gian:** 30 giây

**Thao tác:**

1. Chọn **Mô hình & đánh giá** trong sidebar; giữ chế độ **So sánh mô hình**.
2. Chỉ nhanh vào hạng 1 của Stacking và biểu đồ gồm năm mô hình.
3. Chọn **Chất lượng mô hình**.
4. Giữ ma trận ở **Ngưỡng 30%** và chỉ vào Recall cùng Precision Tiêu cực.

**Lời nói:**

> Bảng xếp hạng cho thấy Stacking đứng đầu năm mô hình theo 5-fold cross-validation, với Macro F1 là 0,5619. Trên final test, ngưỡng Tiêu cực 30% tăng Recall từ 26,32% lên 35,09%, nhưng Precision giảm từ 53,57% xuống 44,44%. Đây là đánh đổi khi ưu tiên phát hiện review tiêu cực.

## Kết thúc demo và buổi trình bày

> Qua demo, nhóm đã thể hiện được toàn bộ luồng từ insight doanh nghiệp, tiền xử lý, dự đoán review cho đến giải thích và đánh giá mô hình. Phần demo của em đến đây là kết thúc.
>
> Nhóm 4 xin chân thành cảm ơn thầy và các bạn đã theo dõi. Nhóm xin sẵn sàng trả lời câu hỏi.

**Thao tác:** Giữ nguyên trang Mô hình & đánh giá. Không chuyển lại PowerPoint.

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
