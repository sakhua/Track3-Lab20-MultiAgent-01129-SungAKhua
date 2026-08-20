# Benchmark Report

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| single-agent baseline | 12.83 | 0.0004 |  |  | 0% |  |
| multi-agent | 23.46 | 0.0015 |  | 20% | 0% |  |

## Comparison

- Fastest run: **single-agent baseline** (12.83s)
- Cheapest run: **single-agent baseline** ($0.0004)

## Failure mode analysis

Cả hai run đều hoàn thành (`failure_rate = 0%`), nhưng citation coverage cho thấy 2 vấn đề:

1. **Baseline: 0% coverage.** Không có bước retrieval, nên model chỉ trả lời từ kiến thức nội tại — không có gì để trích dẫn hay kiểm chứng.
2. **Multi-agent: chỉ 20% (1/5 nguồn).** Writer được prompt trích dẫn nhưng không bị ép buộc bằng validation, nên bỏ sót nguồn nếu Researcher chọn nguồn ít liên quan tới query.

**Khắc phục:** nối `CriticAgent` (đã có ở `agents/critic.py`) vào workflow làm bước cuối, ép Writer sửa lại nếu citation coverage dưới ngưỡng — hiện chưa nối vì Critic đang là bonus/optional.

