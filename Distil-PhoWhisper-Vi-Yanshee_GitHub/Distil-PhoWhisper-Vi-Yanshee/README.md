# Distil-PhoWhisper tiếng Việt cho Robot Yanshee

**Knowledge Distillation · Vietnamese ASR · Whisper · Google Colab A100 · Raspberry Pi 3B · whisper.cpp**

Dự án xây dựng mô hình nhận dạng giọng nói tiếng Việt gọn nhẹ cho **robot Yanshee chạy Raspberry Pi 3B (ARMv7l, 32-bit, RAM 1 GB)**. Mô hình Student được khởi tạo từ `vinai/PhoWhisper-base` bằng cách giữ nguyên Encoder và rút Decoder từ **6 xuống 2 lớp**; sau đó chưng cất tri thức từ `vinai/PhoWhisper-large` trên GPU A100, lượng tử hóa INT8 và chuyển sang **GGML Q8_0** để suy luận bằng `whisper.cpp` trên CPU ARM.

> **Phạm vi repo:** Ba notebook trong `notebooks/` là **mã nguồn thực tế** dùng để nghiên cứu, train, đánh giá và chuyển đổi mô hình. Trình điều khiển robot qua YanAPI, daemon/VAD và các bản vá C++ cho Raspberry Pi **chưa có trong 3 notebook được cung cấp**; kết quả chạy trực tiếp trên robot dưới đây được ghi theo nhật ký dự án.

## 1. Mã nguồn đã thực hiện

| Notebook | Vai trò | Chức năng |
|---|---|---|
| [`notebooks/dstill_whissper.ipynb`](notebooks/dstill_whissper.ipynb) | Khởi tạo & nghiên cứu | Pipeline audio 16 kHz → Log-Mel 80 bins; kiến trúc Encoder/Decoder; tạo Student (giữ decoder layer 0 và 5); normalize văn bản tiếng Việt; WER/CER; KL + pseudo-label loss; smoke test; sinh dữ liệu TTS Yanshee và lọc pseudo-label. |
| [`notebooks/trainningwhisper.ipynb`](notebooks/trainningwhisper.ipynb) | Huấn luyện & kiểm thử | Train Student với Teacher PhoWhisper-large trên A100; checkpoint; benchmark Teacher/Base/Student; nén CTranslate2 INT8; test OOD Viet-Bud500; train đa miền V2; đóng gói model. |
| [`notebooks/doisangc++.ipynb`](notebooks/doisangc++.ipynb) | Xuất model nhúng | Bổ sung tokenizer assets; sử dụng script chuyển đổi của `whisper.cpp` để tạo GGML FP16; build công cụ `whisper-quantize`; lượng tử hóa sang GGML Q8_0 để dùng trên Pi 3B. |

**Tên notebook được giữ nguyên** như file code gốc, kể cả chính tả của tên tệp, để dễ đối chiếu với quá trình thực nghiệm.

## 2. Kiến trúc & huấn luyện

- **Student:** PhoWhisper-base (~72.6 triệu tham số) → ~55.8 triệu tham số (giảm ~23.2%), Encoder 6 lớp giữ nguyên, Decoder 6 → 2 lớp.
- **Khởi tạo Decoder:** lấy lớp đầu (`0`) và lớp cuối (`5`) từ PhoWhisper-base.
- **Teacher:** `vinai/PhoWhisper-large`, inference không cập nhật trọng số.
- **Loss:** `L_KD = 0.8 × L_KL(T=2, nhân T²) + 1.0 × L_PL` (Cross Entropy với nhãn); Encoder Student bị đóng băng.
- **Chuẩn hóa:** Unicode NFC, chữ thường, loại dấu câu, số thành chữ, gộp alias tên `Yanshee` trước khi đo WER/CER.
- **Đợt 1:** 2.000 mẫu VIVOS-FLEURS (nếu có dữ liệu lệnh Yanshee, notebook trộn bổ sung), batch 16, 4 epochs, checkpoint trên Google Drive.
- **Đợt 2:** thêm 4.000 mẫu VIVOS-FLEURS, 4.000 mẫu Viet-Bud500 train và dữ liệu lệnh Yanshee (~8.200 mẫu tổng theo output notebook), batch 32, 3 epochs, bfloat16 trên NVIDIA A100.

## 3. Kết quả thực nghiệm

### So sánh trong miền (in-domain)

Kết quả từ output đánh giá của `trainningwhisper.ipynb` (VIVOS/FLEURS và Yanshee nếu có trong thư mục dữ liệu):

| Model | Tham số | WER | CER | Độ trễ/câu trên thiết bị benchmark |
|---|---:|---:|---:|---:|
| PhoWhisper-large (Teacher) | 1.543,3M | 3,83% | 1,70% | 118,9 ms |
| PhoWhisper-base | 72,6M | 6,81% | 2,50% | 27,4 ms |
| Student chưa train | 55,8M | 410,00% | 232,15% | 20,7 ms |
| **Distil-PhoWhisper (Student đã train)** | **55,8M** | **12,13%** | **5,44%** | **14,7 ms** |

Độ trễ này là từ phép benchmark trong notebook, **không phải** thời gian chạy thực trên Raspberry Pi. Student sau train nhanh hơn Teacher ~8,09× trong phép đo tương ứng.

### Test ngoài miền (OOD) trên Viet-Bud500

Đánh giá 30 mẫu thuộc tập test của Viet-Bud500 qua model CT2 INT8:

| Thời điểm | WER | CER |
|---|---:|---:|
| Sau đợt train 1 | 21,86% | 10,79% |
| **Sau đợt train đa miền V2** | **11,47%** | **5,35%** |

Các kết quả này là số đo từ output notebook, không phải benchmark mới được chạy khi đóng gói repository.

### Kích thước model

| Định dạng | Dung lượng | Môi trường |
|---|---:|---|
| Hugging Face FP32 | 216,6 MB | Lưu & huấn luyện/đánh giá |
| CTranslate2 INT8 | 59,3 MB | CPU hỗ trợ CTranslate2 (kiểm thử trong Colab) |
| GGML Q8_0 | 60,8 MB | `whisper.cpp` cho Raspberry Pi 3B |

**Lưu ý:** CTranslate2/faster-whisper không phải runtime đã chạy trên Pi ARMv7l 32-bit này. Bản nhúng dùng `whisper.cpp`.

## 4. Triển khai Raspberry Pi 3B

**Thông tin phần cứng (theo nhật ký dự án):** Broadcom BCM2837, Cortex-A53 1.2 GHz, ARMv7l 32-bit, Raspbian Stretch, Python 3.5.3, RAM khả dụng thực tế ~411 MB, không GPU.

Model được đặt tại:

```text
/home/pi/Downloads/whisper.cpp/ggml-distil-phowhisper-q8_0.bin
```

Ví dụ thu âm từ microphone Yanshee:

```bash
arecord -f S16_LE -r 16000 -c 1 -d 4 mic_yanshee.wav
```

Ví dụ chạy bản thực thi `./main` đã được biên dịch và điều chỉnh phù hợp với hệ thống Pi 3B của dự án:

```bash
./main -m ggml-distil-phowhisper-q8_0.bin -f mic_yanshee.wav -l vi -t 3 -bs 1 -bo 1 -nt
```

Theo phép thử thực tế đã được ghi lại: encode ~3.893 ms khi bật NEON; decode ~263 ms (9 token); tổng thời gian xử lý câu âm thanh dài 4 giây ~6,6 giây. Mức RAM quan sát ~135 MB. Quá trình build trên GCC 6.3 cần một số sửa đổi tương thích `__fp16` và giới hạn `audio_ctx = 256`. **Các thay đổi ở mã nguồn C/C++ này chưa được đính kèm trong ba notebook** nên repo hiện tại không cung cấp patch tự động hay binary đã build.

Khi robot chạy bằng pin, nên giới hạn khoảng 2–3 CPU threads hoặc dùng nguồn ổn định để giảm nguy cơ sụt áp.

## 5. Cấu trúc kho mã nguồn

```text
Distil-PhoWhisper-Vi-Yanshee/
├── README.md
├── FILE_INVENTORY.md
├── .gitignore
└── notebooks/
    ├── dstill_whissper.ipynb
    ├── trainningwhisper.ipynb
    └── doisangc++.ipynb
```

**Các artifact có trong Google Drive, không nằm trong gói code này:**

```text
/content/drive/MyDrive/Distil_Whisper_Vi_Yanshee/
├── models/
│   ├── distil_phowhisper_base_init/
│   ├── distil_phowhisper_base_trained/
│   ├── distil_phowhisper_base_ct2_int8/
│   └── whisper_cpp_ggml/
├── data/
│   ├── yanshee_vi_smoke_dataset/
│   └── yanshee_vi_pseudo_labelled/
├── checkpoints_large_run/
├── benchmark_results_ch8.csv
├── ggml-distil-phowhisper-q8_0.bin
└── Yanshee_Distil_PhoWhisper_INT8_Package.zip
```

Các notebook dùng đường dẫn cố định dưới Google Drive như trên. Để tái chạy cần mount Drive, thiết lập dữ liệu/model tương ứng và cài những thư viện được import (ví dụ: `torch`, `transformers`, `datasets`, `librosa`, `numpy`, `pandas`, `edge-tts`, `ctranslate2`, `faster-whisper`). Các notebook đã được thực nghiệm trong Colab; repo này **chưa đóng băng phiên bản dependency** và không cam kết chạy lại trên Colab mới mà không chỉnh sửa.

## 6. Trình tự tái hiện bằng notebook

1. Mở `notebooks/dstill_whissper.ipynb` trên Colab, chuẩn bị dữ liệu câu lệnh Yanshee, tạo Student khởi tạo và pseudo-label.
2. Chạy `notebooks/trainningwhisper.ipynb` với A100 để train/benchmark Student, mở rộng dữ liệu đa miền và xuất CTranslate2 INT8.
3. Chạy `notebooks/doisangc++.ipynb` trong môi trường có các công cụ `whisper.cpp` phù hợp với script notebook để xuất `ggml-distil-phowhisper-q8_0.bin`.
4. Chuyển GGML Q8_0 sang Pi 3B, dùng bản `whisper.cpp` đã chỉnh sửa/build cho ARMv7l và thử thu âm với `arecord`.

> Notebook chuyển đổi dựa vào script `models/convert-h5-to-ggml.py` và công cụ `whisper-quantize` của phiên bản `whisper.cpp` tương ứng với thời điểm thực nghiệm; các lệnh có thể cần điều chỉnh khi dùng phiên bản mới.

## 7. Trạng thái phát triển

- [x] Tạo Student 2 Decoder layers từ PhoWhisper-base.
- [x] Tạo/lọc dữ liệu lệnh Yanshee; xây dựng loss KD và đánh giá WER/CER.
- [x] Train đợt 1 và train tiếp đa miền V2 trên A100.
- [x] Xuất CTranslate2 INT8, GGML Q8_0.
- [x] Kiểm thử nhận dạng giọng nói trực tiếp trên Pi 3B (theo nhật ký dự án).
- [ ] Đưa patch `whisper.cpp`/script build Raspberry Pi lên repo.
- [ ] Viết daemon nhận dạng liên tục, giữ model trong RAM; tích hợp VAD.
- [ ] Tích hợp command parser tiếng Việt và SDK `YanAPI` để điều khiển chuyển động có kiểm soát.
- [ ] Đánh giá thêm nhiều giọng Bắc/Trung/Nam và đo RTF/độ trễ đầu-cuối trên robot.

## 8. Tài liệu tham khảo & phát hành

- [Distil-Whisper — arXiv:2311.00430](https://arxiv.org/abs/2311.00430)
- [huggingface/distil-whisper](https://github.com/huggingface/distil-whisper)
- [VinAI PhoWhisper-base](https://huggingface.co/vinai/PhoWhisper-base) / [PhoWhisper-large](https://huggingface.co/vinai/PhoWhisper-large)
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [GitHub repo dự án](https://github.com/phuccode25/LLM)
- [GitHub Release v1.0](https://github.com/phuccode25/LLM/releases/tag/v1.0) — theo liên kết dự án đã cung cấp; model không đi kèm gói source ở đây.

---

**Ghi chú về quyền sử dụng:** Trước khi phát hành public model hoặc dataset, kiểm tra license của PhoWhisper, các bộ dữ liệu nguồn và thư viện phụ thuộc. Chưa cấp giấy phép mới cho mã nguồn dự án trong bản đóng gói này.
