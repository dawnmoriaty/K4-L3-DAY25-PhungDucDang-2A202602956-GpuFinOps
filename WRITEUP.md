# Báo Cáo Kỹ Thuật FinOps — Tối Ưu Hóa Chi Phí GPU (NimbusAI)

> **Tác giả:** Phùng Đức Đăng (MSSV: 2A202602956)  
> **Khóa học:** VinUni AI20k · AICB Phase 2 · Track 2 (Infrastructure) · Day 25  
> **Dự án:** NimbusAI LLM Cost Optimization & FinOps Governance  

---

## 1. Tóm Tắt Điều Hành (Executive Summary)

NimbusAI đối mặt với chi phí hạ tầng GPU tăng phi mã do mở rộng dịch vụ LLM. Áp dụng phương pháp luận FinOps chuẩn hóa theo đơn vị **`$/1M-token`** (đo lường chi phí trên giá trị sản lượng thực nhận thay vì chi phí thuê chỗ `$/GPU-giờ`), dự án đã đạt được kết quả cắt giảm chi phí vượt bậc:

- **Tổng chi phí hàng tháng:** Giảm từ **$27,133** (Baseline) xuống **$14,626** (Optimized).
- **Mức tiết kiệm dự phóng:** **$12,507/tháng** — đạt tỷ lệ cắt giảm **46.1%** (đạt ngưỡng cam kết 40–95%).
- **Đơn giá suy luận (Inference Unit Economics):** Giảm từ **$6.488 / 1M-token** xuống **$1.126 / 1M-token** (tiết kiệm **82.6%**).
- **Tính bền vững (Sustainability):** Tiêu thụ trung bình **0.24 Wh/query**, phát thải **0.091 gCO₂e/query**. Xác định **`europe-north1`** (Na Uy) là vùng hạ tầng vừa rẻ nhất ($0.09/kWh) vừa sạch nhất (30 gCO₂/kWh) nhờ nguồn thủy điện tái tạo.

---

## 2. Phân Tích Chi Tiết Các Đòn Bẩy Tiết Kiệm (Savings Breakdown by Lever)

Hệ thống tối ưu hóa kết hợp 4 đòn bẩy chiến lược từ M1 đến M4:

| Đòn bẩy (Lever) | Tiết kiệm hàng tháng (USD) | Tỷ trọng đóng góp | Cơ chế tác động |
|---|---|---|---|
| **Purchasing Strategy (Spot / Reserved)** | **$10,040** | **80.3%** | Chuyển đổi các job ngắt quãng sang Spot + Checkpoint; cam kết Reserved 3-yr cho workload liên tục vượt điểm hòa vốn (55%). |
| **Inference Levers (Cascade / Cache / Batch)** | **$1,212** | **9.7%** | Định tuyến cascade về model nhỏ rẻ hơn 15–37×; chiết khấu 90% prompt caching; chiết khấu 50% batch API cho tác vụ không đồng bộ. |
| **Right-sizing Util-Lies** | **$655** | **5.2%** | Cấu hình lại hoặc hạ cấp phần cứng cho các GPU bị "Util lie" (util 98% nhưng MFU chỉ ~20%). |
| **Kill Idle GPUs** | **$600** | **4.8%** | Tắt tự động các GPU không có workload phát sinh (cắt giảm 8 giờ idle/ngày của `gpu-h100-5`). |
| **TỔNG CỘNG** | **$12,507** | **100.0%** | **Cắt giảm 46.1% ngân sách hạ tầng GPU** |

![Biểu đồ Waterfall Tối ưu Chi phí](file:///mnt/win_d/VinUni/Lab-25/K4-L3-DAY25-PhungDucDang-2A202602956-GpuFinOps/outputs/savings.png)

---

## 3. Bản Chất Kỹ Thuật: "GPU-Util Lie" & Mô Hình Roofline

### 3.1. Hiện tượng "GPU-Util Lie"
Qua kiểm toán tại M1, công cụ phát hiện 2 GPU rơi vào trạng thái đánh lừa giám sát (`util >= 90%` nhưng `MFU < 30%`):
- **`gpu-h100-4`:** `gpu_util_pct = 98.2%`, nhưng **MFU chỉ đạt 0.194 (19.4%)** và MBU đạt 0.207.
- **`gpu-a10g-1`:** `gpu_util_pct = 96.9%`, nhưng **MFU chỉ đạt 0.268 (26.8%)** và MBU đạt 0.302.

### 3.2. Nguyên nhân gốc rễ (Root Cause Mechanism)
1. **Bản chất đo lường của `nvidia-smi`:** Chỉ số `gpu_util_pct` phản ánh tỷ lệ phần trăm thời gian trong chu kỳ lấy mẫu mà một hoặc nhiều kernel đang chạy trên GPU (xung nhịp clock bận). Nó **hoàn toàn không đo lường số lượng phép toán FLOPs thực tế** được các nhân Tensor Core thực hiện.
2. **Memory Stall trong pha Decode của LLM:**
   - Dựa trên **Roofline Model**, điểm bão hòa (Ridge Point) của NVIDIA H100 (BF16) là khoảng **~295 FLOP/byte**.
   - Pha **Prefill** (tính toán song song toàn bộ prompt) có cường độ tính toán (Arithmetic Intensity) cao (~455 FLOP/byte) $\rightarrow$ **Compute-bound** (tận dụng tối đa Tensor Cores, MFU cao).
   - Pha **Decode** (sinh từng token tuần tự theo cơ chế autoregressive) có cường độ tính toán cực thấp (~1–2 FLOP/byte) $\rightarrow$ **Memory-bound**.
   - Với các workload inference decode mà không có batching động (continuous batching / PagedAttention), GPU H100 liên tục bị nghẽn băng thông bộ nhớ HBM3. Các nhân tính toán rơi vào trạng thái "đói dữ liệu" (stall chờ nạp weights từ bộ nhớ), clock vẫn quay (báo 98% bận) nhưng năng lực xử lý thực chỉ đạt 19.4%.
3. **Tác hại tài chính:** Doanh nghiệp chi trả $2.50/giờ cho một cỗ máy H100 nhưng chỉ thu về giá trị tương đương một GPU phân khúc thấp hơn nhiều, lãng phí hơn 80% chi phí thuê phần cứng.

---

## 4. Giải Đáp Toàn Bộ Câu Hỏi Phân Tích (Guide.md §4.3, §5.4, §6.4, §7.4)

### 4.1. Phân tích M1 — Kiểm toán hiệu quả GPU (§4.3)
1. **GPU có GPU-Util cao nhất:** `gpu-h100-4` (98.2%), nhưng MFU chỉ đạt 19.4% (đỉnh peak 990 TFLOPs FP16 nhưng thực tế chỉ đạt ~192 TFLOPs).
2. **Nguyên nhân chênh lệch Util 98% vs MFU 20%:** Như phân tích ở Mục 3, do kernel decode bị nghẽn băng thông bộ nhớ (memory stall) và kernel launch overhead, khiến GPU bận xung nhịp nhưng rỗng phép tính.
3. **Lãng phí Idle:** `gpu-h100-5` để không 8 tiếng/ngày, lãng phí **$20.00/ngày = $600/tháng**, chiếm 4.1% tổng ngân sách tối ưu.

### 4.2. Phân tích M2 — Đòn bẩy chi phí Inference (§5.4)
1. **Đòn bẩy tiết kiệm lớn nhất:** **Cascade Routing**. Model `large` có giá $3.00 in / $15.00 out per 1M tokens, trong khi model `small` chỉ $0.20 in / $0.40 out (rẻ hơn 15 đến 37.5 lần). Chuyển 70-80% truy vấn đơn giản sang model nhỏ giúp triệt tiêu phần lớn chi phí gốc.
2. **Bản chất `discount_stack = 0.05`:** Chiết khấu có tính chất nhân dồn:
   $$\text{Hệ số} = (1 - 0.50_{\text{batch}}) \times (1 - 0.90_{\text{cache}}) = 0.50 \times 0.10 = 0.05 \quad (\text{tiết kiệm } 95\%)$$
3. **Khi nào KHÔNG nên dùng Batch API:** Các ứng dụng tương tác thời gian thực (real-time chat, autocomplete, trợ lý giọng nói) yêu cầu độ trễ tính bằng mili-giây, trong khi Batch API có SLA phản hồi từ 10 phút đến 24 giờ.

### 4.3. Phân tích M3 — Chiến lược mua GPU (§6.4)
1. **Các job đề xuất dùng Spot:** `job-train-llm`, `job-train-embed`, `job-finetune`, `job-dev-sandbox`, `job-batch-eval` vì có đặc tính `interruptible = 1` và chạy dưới 24h/ngày.
2. **Ý nghĩa "Effective hours" cao hơn "Job hours" ở Spot:** Khi chạy Spot có xác suất thu hồi máy (~5%/giờ), hệ thống phải chịu chi phí ghi checkpoint định kỳ (overhead 3%) và thời gian tính toán lại (rework ~0.5h/lần ngắt). Do đó tổng số giờ máy thực chạy sẽ cao hơn, nhưng đơn giá Spot rẻ hơn 40-60% nên tổng chi phí thực vẫn giảm mạnh.
3. **Đánh giá rủi ro Reserved vs On-demand:** `job-infer-search` trên GPU L4 có duty cycle 18h/ngày (75% > 55% break-even) được gợi ý mua Reserved 3 năm. Tuy nhiên nếu đây là dự án mới chạy thử nghiệm (thời gian ngắn < 30 ngày), việc cam kết 3 năm sẽ tạo ra gánh nặng chôn vốn nếu dự án dừng sớm.

### 4.4. Phân tích M4 — Phân bổ chi phí & chuẩn FOCUS (§7.4)
1. **Team tốn kém nhất:** Team `assistant` tiêu tốn **$2.59/ngày** (30.6%), theo sau là `search` ($2.49/ngày).
2. **Tag coverage & Chargeback:** Tỷ lệ phủ tag đạt **92%**, vượt xa ngưỡng tối thiểu 80%. Hệ thống sẵn sàng kích hoạt **Chargeback Gate** (`chargeback_ready = True`) để khấu trừ ngân sách minh bạch.
3. **Tầm quan trọng của FOCUS:** Khi vận hành trên môi trường đa đám mây (AWS, GCP, Azure, Lambda Labs, RunPod), mỗi nhà cung cấp có cột dữ liệu và cách tính phí riêng biệt. FOCUS cung cấp lược đồ chuẩn hóa mở giúp FinOps đối soát nhất quán.

---

## 5. Kết Quả Triển Khai Phần Mở Rộng "Your Turn" (≥ 2 Extensions)

### 5.1. Extension 1: Nâng cấp chính sách chọn Tier (`recommend_tier`)
- **Vấn đề của logic cũ:** Chỉ xét `duty_cycle` và `interruptible`, dễ dẫn đến việc đề xuất hợp đồng cam kết dài hạn (Reserved 3 năm) cho những công việc thử nghiệm chỉ chạy ngắn ngày (ví dụ 5–15 ngày).
- **Cải tiến triển khai:** Bổ sung tham số `job_days` và `gpu_type` vào [finops/pricing.py](file:///mnt/win_d/VinUni/Lab-25/K4-L3-DAY25-PhungDucDang-2A202602956-GpuFinOps/finops/pricing.py):
  ```python
  if job_days is not None and job_days < 30:
      return "on_demand"  # Tránh cam kết khóa vốn 3 năm cho job tạm thời
  ```
- **Đo lường & Kiểm thử:** Đã viết unit test tự động tại `tests/test_extensions.py::test_recommend_tier_with_job_days` (PASS 100%).

### 5.2. Extension 4: Phân tích ngân sách Reasoning (Reasoning Energy & Cost Budget)
- **Phương pháp đo lường:** Tách biệt và đối chiếu 2,400 request trong [data/token_usage.csv](file:///mnt/win_d/VinUni/Lab-25/K4-L3-DAY25-PhungDucDang-2A202602956-GpuFinOps/data/token_usage.csv) giữa nhóm `is_reasoning = 1` và `is_reasoning = 0`.
- **Bảng số liệu thực nghiệm:**

| Nhóm truy vấn | Số lượng Request | Sản lượng Token | Chi phí ($/ngày) | Tổng điện năng (Wh) | Điện năng / Query |
|---|---|---|---|---|---|
| **Reasoning (`is_reasoning=1`)** | **201 (8.4%)** | 1,241,156 (16.5%) | $1.40 | **29,787.74 Wh (94.0%)** | **148.20 Wh/query** |
| **Normal (`is_reasoning=0`)** | **2,199 (91.6%)** | 6,291,871 (83.5%) | $7.09 | **1,887.56 Wh (6.0%)** | **0.86 Wh/query** |
| **Tỷ lệ chênh lệch** | — | — | — | **Reasoning ngốn 94% điện** | **Gấp 172.3 lần / query** |

- **Insight cốt lõi:** Dù chỉ chiếm 8.4% tổng số lượt gọi, các truy vấn suy luận sâu (Reasoning / Chain-of-Thought) tiêu thụ tới **94% tổng năng lượng điện** của toàn hệ thống (mỗi query tốn tới 148.2 Wh, đủ thắp sáng bóng đèn LED trong 15 giờ).
- **Quy tắc đề xuất (Routing Governance):** Đưa ra cổng phân loại độ phức tạp (Classifier Gate). Chỉ cho phép bật `is_reasoning` khi câu hỏi thuộc danh mục toán/logic phức tạp hoặc khi người dùng sẵn sàng chi trả gói cước cao cấp.

### 5.3. Extension 3 (Bổ sung): Kinh tế học Prompt Caching (`cache_is_worth_it`)
- **Triển khai hàm:** `pricing.cache_is_worth_it()` tính toán điểm hòa vốn giữa chi phí lưu cache prefix (`write_cost_per_m`) và tổng tiền tiết kiệm được qua các lần đọc (`avg_cache_reads * read_price * discount`).
- **Kết quả kiểm thử:** Đạt 100% pass tại `tests/test_extensions.py::test_cache_is_worth_it`.

---

## 6. Kế Hoạch Hành Động Đề Xuất Cho NimbusAI (Actionable Playbook)

Nếu là FinOps Lead tại NimbusAI, 3 hành động triển khai ngay trong tháng đầu tiên:
1. **Thiết lập đàm phán hợp đồng hạ tầng (Tuần 1):**
   - Chuyển toàn bộ 5 job training/eval ngắt quãng sang Spot Instances kết hợp lưu checkpoint vào S3/GCS mỗi 30 phút.
   - Ký hợp đồng Reserved 1-3 năm cho 3 cụm phục vụ inference chạy 24/7 (`job-infer-chat`, `job-infer-rag`, `job-infer-search`). Mang lại khoản tiết kiệm ngay lập tức **$10,040/tháng**.
2. **Chuẩn hóa API Gateway & Inference Stack (Tuần 2):**
   - Triển khai Semantic Router để kích hoạt **Cascade Routing** (chuyển 70% request về model nhỏ).
   - Bật Prompt Caching trên vLLM/SGLang cho các đoạn system prompt cố định.
   - Áp dụng hạn ngạch chặt chẽ cho Reasoning tokens.
3. **Hoàn thiện FinOps Governance & Di dời Data Center (Tuần 3–4):**
   - Ban hành chính sách gắn tag bắt buộc (`team`, `project`), duy trì tag coverage > 90% để thực thi Chargeback nội bộ.
   - Di chuyển các cụm huấn luyện mô hình sang vùng **`europe-north1` (Na Uy)** để vừa giảm tiền điện ($0.09/kWh) vừa giảm 92% lượng phát thải carbon so với `us-east-1`.
