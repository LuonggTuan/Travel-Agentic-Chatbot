from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2")
model.max_seq_length = 2048
sentences_1 = ["Tôi có thể nhận hóa đơn cho chuyến bay đã đặt không?", "Lợi ích của giấc ngủ"]
sentences_2 = ["1. **Tôi có thể nhận hóa đơn cho chuyến bay đã đặt không?**  \nVN Airline có thể gửi lại xác nhận đặt chỗ (E‑ticket) miễn phí trong vòng 90 ngày sau khi vé đã được sử dụng hoàn toàn. Sau 90 ngày, có thể áp dụng phí theo quy định. E‑ticket có thể được dùng như hóa đơn tại hầu hết quốc gia.\n2. **Tôi có cần xác nhận lại chuyến bay không?**  \nKhông. VN Airline không yêu cầu xác nhận lại.\n3. **Tôi có thể kiểm tra giá vé và tình trạng chỗ mà không cần đặt không?**  \nCó. Bạn có thể kiểm tra thông tin chuyến bay mà không cần thanh toán hay hoàn tất đặt chỗ.", 
               "Giấc ngủ giúp cơ thể và não bộ nghỉ ngơi, hồi phục năng lượng và cải thiện trí nhớ. Ngủ đủ giấc giúp tinh thần tỉnh táo và làm việc hiệu quả hơn."]
query_embedding = model.encode(sentences_1)
doc_embeddings = model.encode(sentences_2)
similarity = query_embedding @ doc_embeddings.T
print(similarity)

'''
array([[0.66212064, 0.33066642],
       [0.25866613, 0.5865289 ]], dtype=float32)
'''