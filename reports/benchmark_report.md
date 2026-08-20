# Benchmark Report

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| single-agent baseline | 14.87 | 0.0004 | 1.0 |  | 0% |  |
| multi-agent | 36.76 | 0.0015 | 2.5 | 100% | 0% |  |

## Comparison

- Fastest run: **single-agent baseline** (14.87s)
- Cheapest run: **single-agent baseline** ($0.0004)

## Failure mode analysis

Cả hai run hoàn thành không lỗi (`failure_rate = 0%`), nhưng quality score (chấm bởi Gemini, độc lập với OpenAI đang chạy agent) đều thấp so với thang 10 — 1.0 cho baseline, 2.5 cho multi-agent — vì rubric của corpus (`gold_coverage_points`) đòi hỏi phân tích sâu, phản biện, và đề xuất kiểm chứng được, trong khi cả hai câu trả lời chỉ tổng hợp thông tin ở mức mô tả.

1. **Baseline: citation coverage trống (không trích được nguồn nào).** Không có bước retrieval, model chỉ trả lời từ kiến thức nội tại nên không có gì để đối chiếu — đây là nguyên nhân chính khiến quality thấp nhất.
2. **Multi-agent: quality vẫn chỉ 2.5/10 dù citation coverage đạt 100%.** Trích dẫn đủ nguồn không đồng nghĩa phân tích đủ sâu — Writer hiện chỉ tổng hợp `research_notes`/`analysis_notes` thành văn xuôi, chưa được prompt để bám sát từng `gold_coverage_points` cụ thể (vd. phải có phản biện, phải chỉ ra điều kiện single-agent tốt hơn).

**Khắc phục:** (a) nối `CriticAgent` (đã có ở `agents/critic.py`) vào workflow làm bước cuối để ép Writer sửa lại khi thiếu trích dẫn hoặc thiếu phản biện; (b) truyền `gold_coverage_points` của topic vào prompt của Writer để nó bám sát rubric thay vì viết tự do.
