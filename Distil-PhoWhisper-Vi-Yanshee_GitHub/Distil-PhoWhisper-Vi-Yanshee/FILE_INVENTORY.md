# Thống kê code dự án Distil-PhoWhisper-Vi-Yanshee

## Những file code được cung cấp

| STT | File gốc | Nội dung cụ thể | Tạo ra / Sử dụng |
|---|---|---|---|
| 1 | `dstill_whissper.ipynb` | Sinh waveform/log-Mel, rút Decoder 6→2, thiết lập loss KL+CE, normalize & đánh giá WER/CER, tạo TTS câu lệnh Yanshee, pseudo-label bằng Teacher, lọc lỗi, smoke test | `models/distil_phowhisper_base_init/`, `data/yanshee_vi_smoke_dataset/`, `data/yanshee_vi_pseudo_labelled/`; thực nghiệm model student |
| 2 | `trainningwhisper.ipynb` | Train VIVOS-FLEURS, benchmark 4 model, chuyển sang CTranslate2 INT8, test Viet-Bud500 OOD, train V2 trên 8.200 mẫu, cập nhật ZIP model | `checkpoints_large_run/checkpoint_a100_latest.pt`, `models/distil_phowhisper_base_trained/`, `models/distil_phowhisper_base_ct2_int8/`, `benchmark_results_ch8.csv`, `Yanshee_Distil_PhoWhisper_INT8_Package.zip` |
| 3 | `doisangc++.ipynb` | Bổ sung assets tokenizer, convert Hugging Face → GGML FP16 bằng script whisper.cpp, build công cụ quantize rồi tạo Q8_0 | `models/whisper_cpp_ggml/ggml-model.bin`, `ggml-distil-phowhisper-q8_0.bin` |

## Đối chiếu kết quả quan trọng

| Chỉ số | Kết quả | Bằng chứng |
|---|---:|---|
| Student | 55.8M tham số; Decoder 2 lớp | Output notebook `dstill_whissper` |
| Train đợt 1 | Loss trung bình epoch 4: 0.5651 | Output cell huấn luyện `trainningwhisper` |
| In-domain | WER 12.13%; CER 5.44%; latency 14.7ms/câu | Output cell benchmark `trainningwhisper` |
| OOD Viet-Bud500 trước V2 | WER 21.86%; CER 10.79% | Output cell test OOD `trainningwhisper` |
| OOD Viet-Bud500 sau V2 | WER 11.47%; CER 5.35% | Output cell multi-domain `trainningwhisper` |
| CT2 INT8 | 59.3 MB | Output cell convert `trainningwhisper` |
| GGML Q8_0 | 60.8 MB | Output cell quantize `doisangc++` |
| Test trực tiếp Pi 3B | ~6.6s cho âm thanh 4s; ~135 MB RAM | Nhật ký dự án do tác giả cung cấp; không có mã/test log Pi trong các notebook tải lên |

## Chưa có trong 3 notebook tải lên

- Bản sửa `ggml.h`, `ggml.c`, `examples/common-ggml.cpp`, `main.cpp` đã build trên Pi 3B.
- Script/daemon Python 3.5 luôn nghe, giữ model trong RAM, tích hợp VAD.
- Mã command parser / `YanAPI.start_play_motion(...)`.
- File model nhị phân và CSV benchmark thực tế (chỉ mới có mã tạo ra chúng và output notebook).

Kết luận: **3 notebook hiện tại ghi lại pipeline tạo và tối ưu model. Phần mã triển khai robot trực tiếp cần bổ sung riêng** để repository có thể tái hiện đầy đủ đầu-cuối.
