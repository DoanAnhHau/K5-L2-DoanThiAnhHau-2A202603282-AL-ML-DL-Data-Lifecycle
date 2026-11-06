# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn
**Runtime Colab:** 6/11/2026  
**Python / PyTorch / Ultralytics:**  Python 3.13.18 / PyTorch 2.11.0+cpu / Ultralytics 8.4.145  
**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`  
**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`  
**Thay đổi so với notebook nguồn:** Không.

## 1. Phân loại ảnh

**Artifact tham chiếu:** `day1_lab_outputs/classification_predictions.json` và `day1_lab_outputs/visuals/classification_top5.png`.  
Theo yêu cầu, phần dưới chỉ dùng ảnh `sample_id = traffic`, `rank = 1`.

1. Record hạng 1: `class_id=468`; `class_name=cab`; `rank=1`; `score=0.510915`; `taxonomy_name=ImageNet-1K`.
2. Đây là dự đoán cho **cả bức ảnh**. Classification trả về nhãn của ảnh đầu vào, không chỉ ra một vùng hay một vật thể cụ thể bằng bounding box. Model chọn `cab` là lớp đứng đầu trong các lớp của ảnh giao thông này.
3. Danh sách lớp đến từ taxonomy gắn với checkpoint, ở đây là `ImageNet-1K`; model chọn lớp trong danh sách đó chứ không tự nghĩ ra tên lớp mới.
4. `class_id` là mã lớp ổn định để máy đối chiếu; `class_name` dễ đọc cho người; `taxonomy_name` cho biết mã và tên đó thuộc bộ lớp nào. Chỉ ghi tên có thể gây nhầm khi các taxonomy khác nhau dùng tên giống nhau hoặc cách đánh số khác nhau.
5. Guideline cần định nghĩa rõ mục tiêu phân loại: chủ thể/chủ đề chính là gì, ưu tiên vật thể nào khi có nhiều vật, và cách xử lý ảnh có nhiều chủ đề hoặc không có lớp phù hợp. Mọi annotator cần áp dụng cùng một quy tắc chọn một nhãn.
6. `score=0.510915` chỉ thể hiện mức độ model nghiêng về dự đoán `cab` so với các lớp khác; nó không chứng minh ảnh thật sự thuộc lớp đó. Ground truth phải được gán theo ảnh và guideline rồi được QC, không lấy trực tiếp từ score.

## 2. Phát hiện vật thể

**Artifact tham chiếu:** `day1_lab_outputs/detection_predictions.json` và `day1_lab_outputs/visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen` (kích thước ảnh: 640 × 427 pixel). Chọn prediction `person` có score cao nhất.

1. Record: `class_name=person`; `score=0.912625`; `bbox_xyxy=[385.33, 69.24, 498.92, 348.92]`; `bbox_width=113.58`; `bbox_height=279.68`.
2. Tọa độ tính từ góc trên-trái ảnh, trục x hướng sang phải và trục y hướng xuống. Box bắt đầu tại góc trên-trái khoảng `(385.33, 69.24)` và kết thúc tại góc dưới-phải khoảng `(498.92, 348.92)`. Theo JSON, box rộng khoảng `113.58` pixel và cao `279.68` pixel.
3. Trong JSON đã xuất ở ngưỡng `0.35`, ảnh `kitchen` có **11 prediction**. Trong 11 record này, **6 prediction có score ≥ 0.60**. Đây là so sánh `0.35` với `0.60`; artifact hiện có không cho biết số lượng ở ngưỡng `0.20`.
4. Hạ ngưỡng thường giữ lại nhiều box hơn, kể cả box sai, trùng hoặc không rõ. Reviewer phải rà soát thêm false positive, box trùng, lớp sai và đối tượng bị bỏ sót; không chấp nhận prediction chỉ vì score cao hay thấp.
5. **Quy tắc box:** vẽ box sát nhất có thể quanh toàn bộ phần nhìn thấy của một đối tượng thuộc lớp cần gán; không bao gồm vật khác và không tự kéo box ra ngoài khung ảnh để đoán phần bị khuất/cắt.
6. Ở mép trái ảnh có một phần người bị cắt khỏi khung hình (thể hiện trong PNG detection như một prediction `person`). Nên **escalate lên Lab Coach** nếu guideline chưa nói rõ có gán đối tượng chỉ nhìn thấy một phần hay không, vì quyết định này ảnh hưởng tính nhất quán giữa annotator. Nếu guideline đã quy định cách xử lý vật bị cắt mép thì làm theo quy tắc đó.

## 3. Phân đoạn theo từng đối tượng

**Artifact tham chiếu:** `day1_lab_outputs/segmentation_predictions.json` và `day1_lab_outputs/visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`, chọn record đầu tiên.

1. Record: `instance_id=kitchen-001`; `class_name=person`; `score=0.899318`; `polygon_point_count=348`; 5 điểm đầu của `polygon_xy` là `[[446, 70], [445, 71], [444, 71], [443, 72], [442, 72]]`.
2. Box chỉ cho biết vùng chữ nhật bao quanh vật thể. Polygon/mask mô tả đường biên theo hình dạng vật thể, nên thể hiện được dáng người và vùng pixel thuộc người thay vì tô cả vùng nền nằm trong một hình chữ nhật.
3. `instance_id` phân biệt từng đối tượng riêng trong một ảnh và liên kết polygon với record tương ứng. Nó không phải class ID (không biểu thị lớp như `person`) và không phải tracking ID (không theo dõi cùng một đối tượng qua nhiều khung hình/video).
4. **Quy tắc mask:** đi theo ranh giới nhìn thấy giữa vật thể và nền, chỉ tô pixel thuộc vật thể; không tô lấn sang vật kế bên và không tự vẽ phần bị che khuất như thể nhìn thấy.
5. PNG segmentation cho thấy bàn và nhiều vật dụng trên bàn chạm/chồng lấn, nên có những đoạn khó phân định đường biên mask chính xác. Nếu guideline chưa định nghĩa cách tách phần vật bị che hoặc hai vật chạm nhau, cần **escalate lên Lab Coach** thay vì tự đặt ranh giới; nếu đã có quy tắc thì áp dụng thống nhất.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho mỗi ảnh, theo taxonomy và quy tắc chọn chủ đề chính. | Ảnh `traffic` có nhiều loại xe; prediction hạng 1 là `cab` với score `0.510915`, tiếp theo là `minibus` `0.164284`. Đây là dự đoán, không xác nhận lớp đúng. | Dùng guideline để chọn chủ đề/lớp phù hợp của toàn ảnh; hỏi khi ảnh có nhiều chủ đề mà quy tắc chưa xử lý. | Đối chiếu nhãn với ảnh, taxonomy và guideline; kiểm tra ảnh nhiều chủ đề hoặc nhãn không nhất quán. |
| Phát hiện vật thể | Mỗi instance có class và một bounding box `xyxy`. | JSON `kitchen` có 11 box ở ngưỡng xuất `0.35`; có người bị cắt ở mép trái, và ảnh có nhiều vật nhỏ/chồng lấp. | Gán từng đối tượng thuộc phạm vi, chọn lớp đúng và đặt box quanh phần nhìn thấy; theo guideline với vật bị cắt/che. | Kiểm tra box thiếu/thừa/trùng, lớp sai, độ khít của box và các prediction có score thấp; xác nhận cách xử lý vật bị cắt. |
| Instance segmentation | Mỗi instance có class và mask/polygon riêng, gắn với `instance_id`. | `kitchen-001` là `person`, score `0.899318`, polygon có 348 điểm. PNG cho thấy nhiều vật trên bàn sát nhau, dễ tô lấn hoặc gộp biên. | Vẽ một mask riêng theo đường biên phần nhìn thấy của mỗi instance; không gộp các vật chỉ vì chạm nhau. | Kiểm tra mask có bám biên không, có lấn nền/vật bên cạnh hay bỏ sót vùng nhìn thấy không; kiểm tra tách instance và các vùng che khuất. |

## 5. An toàn dữ liệu

1. Tôi chỉ sử dụng các artifact của bài lab và ảnh COCO công khai, không thêm hoặc đưa thông tin cá nhân/nhạy cảm vào báo cáo hay output.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng xử lý và báo Lab Coach hoặc giáo viên phụ trách để được hướng dẫn.
## 6. Checklist trước khi nộp link repo

Các file dưới đây có trong gói làm việc hiện tại; cần kiểm tra lại sau khi đưa lên GitHub. Trạng thái validation notebook, nội dung repo đã upload và việc nộp LMS không thể xác nhận từ các artifact được cung cấp.

- [x] `REPORT.md` (đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS` 
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output đã kiểm tra
- [x] Đã nộp **link repository** trên LMS .
