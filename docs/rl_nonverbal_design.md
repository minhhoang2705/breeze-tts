# Thiết kế: Post-training bằng RL cho non-verbal vocal events (Breeze TTS 2, tiếng Việt)

> Trạng thái: **bản thiết kế (draft)**. Chưa có code nào được viết theo tài liệu này.
> Phạm vi: checkpoint đã SFT (p1/p2, xem `configs/train_p*_vietnamese_*.json`) và phần cứng 2× H100 80GB.

## 1. Bài toán

Sau SFT, model gặp hai lỗi liên quan đến các tag sự kiện phát âm, ví dụ `(laugh)`, `(cough)`, `(sigh)`, `(clears throat)` hoặc tag tiếng Việt tương đương:

| Lỗi | Mô tả | Cách đo được |
|---|---|---|
| **E1: Không ổn định** | Cùng một prompt nhưng khi sinh mẫu, lúc ra event, lúc không ra, lúc ra lệch vị trí. Model cũng có thể đọc tag thành chữ (ví dụ đọc "cười" thay vì cười). | Tỷ lệ *tag hit rate* trên N mẫu cho mỗi prompt |
| **E2: Nhầm tag** | Yêu cầu `(cough)` nhưng model sinh ra tiếng cười, hoặc ngược lại. | Confusion matrix giữa tag yêu cầu và event phát hiện được |

Mục tiêu là tăng *tag accuracy* và giảm cả confusion lẫn spurious event, **không** làm giảm CER, speaker similarity hay độ tự nhiên.

## 2. Ba nhận định định hướng thiết kế

1. **Rủi ro lớn nhất nằm ở reward, không nằm ở thuật toán.** Whisper-large-v3 và Zipformer tiếng Việt đều là ASR. Chúng được train để *bỏ qua* tiếng cười hay tiếng ho, nên không thể làm reward cho E1/E2. Bản thân `breeze_train/eval_harness.py:131` cũng đang xoá tag trước khi tính WER. Vì vậy cần một **bộ phát hiện non-verbal event** riêng (mục 4), và ASR chỉ đóng vai trò *guardrail*.
2. **RL chỉ khuếch đại những gì model đã làm được đôi khi.** Nếu sinh N=16 mẫu mà không mẫu nào chứa đúng event, DPO và GRPO đều không có tín hiệu để học. Do đó Phase 0 phải đo **pass@N cho từng tag**. Tag nào có pass@N thấp là vấn đề về *dữ liệu SFT* (thiếu mẫu, mất cân bằng, tag bị tokenize kỳ lạ qua text encoder đang bị frozen), và RL không sửa được loại vấn đề này.
3. **Credit assignment.** Một event chỉ kéo dài khoảng 0.3–1.5 s trong một câu dài nhiều giây, mỗi frame lại có 32 codebook. Nếu tính log-prob trên toàn bộ chuỗi thì tín hiệu từ vùng event bị pha loãng. Thiết kế loss vì vậy hỗ trợ **trọng số theo vùng thời gian quanh tag** (mục 5.3).

## 3. DPO hay GRPO/PPO: khuyến nghị theo giai đoạn

| Tiêu chí | Offline DPO (iterative) | Online GRPO / RLOO | PPO |
|---|---|---|---|
| Cần critic/value model | Không | Không | Có (thêm khoảng 1.85B tham số) |
| Rollout trong vòng train | Không, sinh offline theo batch | Có | Có |
| Tích hợp vào code hiện tại | Dễ: subclass `BreezeTrainer` | Trung bình: cần vòng lặp sinh mẫu, train, đồng bộ weight | Khó |
| Ổn định | Cao | Trung bình | Thấp (nhiều siêu tham số) |
| Trần chất lượng | Bị giới hạn vì off-policy | Cao hơn | Cao |
| Trên 2× H100 | Thoải mái, vì ref log-prob tính trước | Khả thi nếu dùng LoRA | Chật |

**Khuyến nghị:** Phase 0 (reward/eval) → Phase 1 (DPO offline) → Phase 2 (DPO iterative, tức on-policy-ish) → Phase 3 (GRPO) **chỉ khi** DPO đã bão hoà trên metric ở mục 7.

Lý do:
- Seed-TTS [1], CosyVoice 2 [2] và Koel-TTS [3] đều cho thấy preference alignment kiểu DPO đem lại cải thiện rõ rệt cho codec-LM TTS với chi phí thấp.
- Cuộc chạy đầu tiên cần trả lời câu hỏi "reward có đúng không?". DPO offline cho phép **soi từng cặp chosen/rejected bằng tai** trước khi train, điều mà GRPO không cho phép.
- GRPO [5] bỏ được critic so với PPO [6], nhưng vẫn cần rollout nhanh và đồng bộ weight với runtime CUDA-graph (`models/fast_streaming.py`). Chi phí kỹ thuật này chỉ đáng bỏ ra khi đã biết reward đáng tin.

## 4. Reward model

### 4.1 Thành phần

| Ký hiệu | Thành phần | Công cụ | Vai trò |
|---|---|---|---|
| `R_evt` | Đúng event, đúng vị trí | **Event detector** (4.2) cùng word timestamps từ ASR | Reward chính cho E1/E2 |
| `P_conf` | Phạt khi phát hiện event *khác* với tag yêu cầu trong cửa sổ của tag | Event detector | Trị E2 |
| `P_spur` | Phạt event xuất hiện ở nơi không có tag | Event detector | Tránh hack "cười khắp nơi" |
| `P_lit` | Phạt khi đọc tag thành chữ | ASR: từ khoá của tag xuất hiện trong transcript | Trị một dạng của E1 |
| `G_cer` | Guardrail CER (text đã xoá tag) | Whisper-large-v3 hoặc Zipformer-vi | Không được giảm khả năng đọc đúng |
| `G_sim` | Guardrail speaker similarity | ECAPA (đã có trong `eval_harness.py:175`) | Với template clone/direction |
| `G_q` | Guardrail chất lượng | UTMOS hoặc DNSMOS (tuỳ chọn) | Chống artefact |

Công thức tổng hợp ban đầu, sẽ hiệu chỉnh ở Phase 0:

```
R = R_evt − α·P_conf − β·P_spur − γ·P_lit − λ·max(0, CER − CER_ref) + μ·(SIM − SIM_ref)
```

`CER_ref` và `SIM_ref` là giá trị của chính checkpoint SFT trên cùng prompt. Guardrail chỉ phạt khi kết quả **tệ hơn baseline**, chứ không thưởng khi đọc tốt hơn, để tránh reward bị CER chi phối.

**Gợi ý thực tế về ASR:** dùng **Zipformer-vi cho CER** vì nó nhanh và chuẩn hơn với tiếng Việt, và dùng **Whisper-large-v3 lấy word timestamps** để căn vị trí tag. Cần kiểm tra lại trên tập dev xem timestamps của Whisper cho tiếng Việt có đủ chính xác không (±200 ms là đủ).

### 4.2 Event detector, thành phần quan trọng nhất

Các lựa chọn, xếp theo thứ tự khuyến nghị:

1. **Fine-tune một classifier âm thanh đã pretrain** (AST [9] hoặc BEATs [10], vốn đã train trên AudioSet [8]). Ontology của AudioSet có sẵn các lớp *Laughter*, *Cough*, *Sigh*, *Throat clearing*, *Breathing*, *Sniff*, … Dữ liệu fine-tune gồm:
   - VocalSound [11], gồm khoảng 21k bản ghi thuộc 6 lớp: laughter, sigh, cough, throat clearing, sneeze, sniff.
   - NonverbalTTS [12], gồm khoảng 17 giờ tiếng Anh với tag căn theo text.
   - **Quan trọng nhất:** các đoạn chứa tag trong chính tập SFT tiếng Việt của bạn, cắt theo cửa sổ thời gian quanh tag.
2. Dùng zero-shot AudioSet classifier không fine-tune. Cách này nhanh để làm baseline nhưng chưa được hiệu chỉnh cho giọng hay tag set của bạn.
3. Dùng CLAP zero-shot. Đây chỉ là phương án dự phòng, vì độ phân biệt giữa các lớp gần nhau như sigh và breath thường kém.

Detector chạy theo **sliding window** (ví dụ 1 s, hop 0.25 s) để cho ra event kèm timestamp, sau đó khớp với vị trí tag lấy từ word timestamps.

**Cổng chất lượng trước khi dùng làm reward:** detector cần đạt macro-F1 ≥ 0.85 trên một tập **gán nhãn thủ công** khoảng 300 clip do *chính Breeze sinh ra*, không phải trên dữ liệu thật. Lý do là reward hacking thường khai thác đúng phần domain gap giữa audio sinh và audio thật.

## 5. Phase 1: DPO offline

### 5.1 Dữ liệu prompt

- Lấy từ tập SFT có tag, cộng thêm một tập prompt tổng hợp: chèn tag vào câu thường với vị trí và loại tag phân bố đều. Cách này buộc model gặp các tổ hợp hiếm, đúng chỗ E2 hay xảy ra.
- Khoảng 3–5k prompt, phủ cả 4 template trong `breeze_infer/templates.py` (`tts_plain`, `tts_instruction`, `ref_clone_tata`, `ref_edit_tata`).
- Giữ riêng một tập held-out để đánh giá (mục 7), không dùng để sinh cặp.

### 5.2 Sinh mẫu và tạo cặp

1. Với mỗi prompt, sinh N = 8–16 mẫu với **đúng cấu hình sampling và CFG dùng khi deploy**.
   - Lưu ý: CFG làm phân phối sinh mẫu khác với policy có điều kiện mà ta tính log-prob. Điều này chấp nhận được với DPO, vì DPO vốn off-policy, nhưng cần ghi lại để ablation.
2. **Lưu codec codes đã sinh** (shape `[frames, 32]`), không chỉ lưu wav. Hiện tại `FastStreamingChunk` (`models/fast_streaming.py:59`) chỉ trả về audio, nên cần thêm hook hoặc trường `codes`. Encode lại wav qua codec sẽ cho ra codes khác với codes model đã thật sự sinh, và khi đó log-prob sẽ sai.
3. Chấm điểm bằng R (mục 4). Cặp được giữ lại khi thoả `R(chosen) − R(rejected) ≥ margin` và **cả hai đều qua guardrail CER**, để model không học cách tránh event bằng cách đọc sai.
4. Ưu tiên **hard negative**:
   - rejected có đúng nội dung text nhưng sai loại event (dành cho E2);
   - rejected thiếu event (dành cho E1).
5. Biến thể kiểu SpeechAlign [4]: dùng **ground-truth codes** của clip SFT có tag làm chosen, và mẫu sai tag của model làm rejected. Cách này hữu ích cho các tag có pass@N thấp.

### 5.3 Loss và cách tích hợp vào code

`BreezeForConditionalGeneration.forward` hiện trả về CE với `reduction='mean'` cho cả batch (`models/breeze.py:1817-1855`). DPO lại cần **tổng log-prob theo từng mẫu**, nên cần viết một hàm riêng:

```python
def sequence_logps(model, batch, frame_weights=None) -> Tensor:  # [B]
    # backbone: log p(cb0_t | ...) tại các vị trí label == audio_token_id
    # depth decoder: log p(cb_k_t | cb_<k, h_t) với k = 1..31; dùng train_idxs[0]
    #   để gom các frame (đã bị flatten qua mọi sample) về lại từng sample
    # frame_weights: [B, T] tuỳ chọn, để tăng trọng số các frame trong cửa sổ tag
```

Các lựa chọn cần ablation:

- (a) Chỉ dùng backbone cb0. Codebook đầu mang nhiều thông tin ngữ nghĩa và prosody, rẻ và ít nhiễu.
- (b) Dùng cb0 cộng depth decoder.
- (c) Dùng (b) kết hợp `frame_weights` tập trung quanh tag.

Loss: `L = L_DPO(β) + λ_sft · NLL(chosen)`. Thành phần NLL trên chosen (như trong RPO [13]) giúp tránh hiện tượng log-prob của cả chosen lẫn rejected cùng giảm.

**Reference log-probs tính trước một lần** rồi lưu vào dataset, nên không cần giữ model ref trên GPU. Nhờ vậy cấu hình p2 trên 2× H100 (khoảng 45 GiB/GPU theo `configs/train_p2_vietnamese_scale3_2xh100.json`) vẫn vừa.

**Policy khuyến nghị:** bắt đầu với **p1 (LoRA)** để rẻ và rollback dễ, với LR khoảng 5e-6 đến 1e-5 và β trong khoảng 0.1–0.5. Chỉ chuyển sang p1b/p2 khi LoRA đã bão hoà.

### 5.4 File dự kiến

| File | Nội dung |
|---|---|
| `breeze_train/rewards/event_detector.py` | Sliding-window detector cho ra event kèm timestamp |
| `breeze_train/rewards/asr.py` | Wrapper Zipformer/Whisper, trả về transcript và word timestamps |
| `breeze_train/rewards/reward.py` | Tổng hợp R, đọc trọng số từ config |
| `scripts/rl_sample.py` | Sinh N mẫu cho mỗi prompt, lưu wav, **codes** và metadata |
| `scripts/rl_build_pairs.py` | Chấm điểm, tạo cặp, tính trước ref log-probs |
| `breeze_train/preference.py` | `PreferenceDataset` và collator, tái dùng `examples.build_example(target_codes=...)` |
| `breeze_train/dpo_trainer.py` | `BreezeDPOTrainer(BreezeTrainer)` cùng `sequence_logps` |
| `train_dpo.py` và `configs/dpo_p1_vietnamese_nv.json` | CLI và config, theo khuôn của `train.py` |
| `breeze_train/eval_harness.py` | Thêm subcommand `score-events`: tag accuracy và confusion matrix |
| `tests/test_sequence_logps.py` | Kiểm tra `−sequence_logps / n_tokens` khớp với `outputs.loss` khi không có trọng số |

## 6. Phase 2–3: on-policy

- **Phase 2, DPO iterative:** sau mỗi vòng, sinh lại mẫu từ checkpoint mới, tạo cặp mới và train tiếp. Thường 2–3 vòng là đủ. Không cần thay đổi hạ tầng.
- **Phase 3, GRPO [5] hoặc RLOO [7]:**
  - GPU0 train (LoRA), GPU1 sinh rollout bằng runtime CUDA-graph. Đồng bộ adapter mỗi k bước.
  - Group size G = 8. Advantage được chuẩn hoá trong nhóm, **không** chia theo độ dài (tham khảo Dr. GRPO [14], vì độ dài audio thay đổi nhiều).
  - Có KL tới ref, dễ làm vì ref chính là model khi tắt adapter.
  - Log-prob cho importance ratio **phải tính lại bằng trainer**, không lấy từ rollout engine, vì kernel CUDA-graph và kernel eager cho ra kết quả số khác nhau.
  - Chỉ triển khai khi DPO iterative không cải thiện thêm sau 2 vòng.

## 7. Đánh giá và tiêu chí dừng

Tập held-out gồm khoảng 300 prompt, phủ đều các tag và template. Mỗi prompt sinh **5 seed**.

| Metric | Mục tiêu so với SFT |
|---|---|
| Tag accuracy (đúng loại và đúng vị trí, trung bình trên các seed) | Tăng ≥ 15 điểm % |
| Confusion rate theo từng cặp tag | Giảm ≥ 50% |
| Độ ổn định: std của hit rate giữa các seed | Giảm |
| Spurious event rate trên câu không có tag | Không tăng |
| CER (đã xoá tag) | Không tăng quá +0.5 điểm tuyệt đối |
| Speaker similarity | Không giảm quá −0.01 |
| Nghe thử A/B mù, khoảng 50 cặp | Tỷ lệ thắng ≥ 60% |

**Dấu hiệu reward hacking cần theo dõi:**
- event dài bất thường;
- lặp event;
- giọng bị "méo" nhưng detector vẫn đếm là event;
- câu bị rút ngắn.

Với mỗi checkpoint, log độ dài audio và phân phối độ dài event.

## 8. Kế hoạch triển khai

| Bước | Việc | Đầu ra / cổng |
|---|---|---|
| P0.1 | Thống nhất tag set và viết đặc tả căn vị trí tag | Tài liệu tag set |
| P0.2 | Xây và hiệu chỉnh event detector | Macro-F1 ≥ 0.85 trên audio do Breeze sinh |
| P0.3 | `score-events` cùng baseline SFT: confusion matrix và **pass@N cho từng tag** | Báo cáo baseline; danh sách tag "cần sửa ở SFT" và tag "RL sửa được" |
| P1.1 | `sequence_logps` cùng unit test | Test pass |
| P1.2 | `rl_sample.py` (lưu codes) và `rl_build_pairs.py` | Bộ cặp đã nghe kiểm 50 cặp |
| P1.3 | `BreezeDPOTrainer`, chạy LoRA, ablation (a)/(b)/(c) | Đạt metric ở mục 7 |
| P2 | 2–3 vòng DPO iterative | Đường cong metric theo vòng |
| P3 | GRPO, tuỳ chọn | Chỉ khi P2 bão hoà |

## 9. Câu hỏi mở cần trả lời trước P0

1. Tag set tiếng Việt cụ thể là gì? Dùng cú pháp `( )` hay `[ ]`, viết bằng tiếng Anh hay tiếng Việt? Mỗi tag có bao nhiêu mẫu trong dữ liệu SFT?
2. Tập SFT có timestamp cho vị trí event không, hay chỉ có vị trí trong text?
3. Khi deploy, CFG và sampling dùng cấu hình nào? `eval_harness` hiện đang dùng `guidance_scale=1.0`.
4. Text encoder đang bị frozen ở mọi policy. Nếu tag được tokenize thành các subword ít nghĩa, liệu cách đơn giản hơn RL có phải là thêm **special token** cho từng tag (cùng với p3) không?

## Tài liệu tham khảo

1. Anastassiou et al., *Seed-TTS: A Family of High-Quality Versatile Speech Generation Models*, arXiv:2406.02430, 2024.
2. Du et al., *CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models*, arXiv:2412.10117, 2024.
3. Hussain et al., *Koel-TTS: Enhancing LLM based Speech Generation with Preference Alignment and Classifier Free Guidance*, EMNLP 2025, arXiv:2502.05236.
4. Zhang et al., *SpeechAlign: Aligning Speech Generation to Human Preferences*, NeurIPS 2024, arXiv:2404.05600.
5. Shao et al., *DeepSeekMath* (giới thiệu GRPO), arXiv:2402.03300, 2024.
6. Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347, 2017.
7. Ahmadian et al., *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs* (RLOO), ACL 2024, arXiv:2402.14740.
8. Gemmeke et al., *Audio Set: An Ontology and Human-Labeled Dataset for Audio Events*, ICASSP 2017.
9. Gong, Chung, Glass, *AST: Audio Spectrogram Transformer*, Interspeech 2021, arXiv:2104.01778.
10. Chen et al., *BEATs: Audio Pre-Training with Acoustic Tokenizers*, ICML 2023, arXiv:2212.09058.
11. Gong, Yu, Glass, *VocalSound: A Dataset for Improving Human Vocal Sounds Recognition*, ICASSP 2022, arXiv:2205.03433.
12. Borisov, Spirin, Diatlova, *NonverbalTTS: A Public English Corpus of Text-Aligned Nonverbal Vocalizations with Emotion Annotations for Text-to-Speech*, SSW 2025, arXiv:2507.13155.
13. Pang et al., *Iterative Reasoning Preference Optimization*, NeurIPS 2024, arXiv:2404.19733.
14. Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective* (Dr. GRPO), arXiv:2503.20783, 2025.
15. Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS 2023, arXiv:2305.18290.
