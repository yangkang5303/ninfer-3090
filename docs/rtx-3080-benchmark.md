# RTX 3080 基准测试与多角度优化

在 RTX 3080(68 SM,20 GiB GDDR6X,sm_86,CUDA 13.3)上,将上游 3090 目标移植运行并完成多角度优化。

## 修复的 bug

| 文件 | 问题 | 修复 |
|---|---|---|
| `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp` | 协作式 MMA 路由上界按 82 SM 硬编码,68 SM 上 prefill 宽 768 时 144 CTA 超出 136 驻留预算,模型加载即抛 `candidate is not legal for exact problem` | 路由上界改为按运行时 `device_sm_count()` 动态推导 |
| `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp` | A16 融合 swiglu 仅注册到 T≤16,prefill 宽 >16 即抛异常 | 在 token 轴上按 32 分块调用小-T 核(镜像 FP8 A16 路线) |
| `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_small_t.cu` | launcher 只注册 2..16 | 扩展到 2..32,与线性小-T 生产 schedule 一致 |

## 瓶颈分析

- **decode 计算受限**:A16 反量化 gemv 在 100% SM 占用、318W/320W 满功耗下,DRAM 带宽仅 ~20%(有效 186 GB/s)。SM 算力被 e4m3/e2m1 标量反量化 FMA 耗尽,非 occupancy/缓存问题(实测 `MinBlocksPerSm 2→4`、L1 绕过均无收益)。
- **prefill 受 micro-MMA 限制**:nvfp4 小-T MMA 每次 launch 全量重读权重,分块方案本质受限。
- **根因**:`nvfp4` 在 sm_86 上走 A16 兼容降级路线(SIMT 反量化);`groupwise-int` 才是 fork 首调优的 q4/q5 MMA 路线。

## 多角度优化结果

| 指标 | nvfp4 A16(原) | groupwise-int(推荐) | 提升 |
|---|---:|---:|---:|
| 权重显存 | 16.03 GiB | **15.25 GiB** | −0.8 GiB |
| prefill(长 prompt) | 19.4 tok/s | **558–636 tok/s** | ~29× |
| decode(贪心) | 11.7 tok/s | **33.7 tok/s** | 2.9× |
| decode(MTP3) | 33.0 tok/s | **69.5 tok/s** | 2.1× |

投机解码:MTP3 最优(87–91% 接受率);draft=5 反降至 67.6 tok/s(接受率 73%)。

## KV 量化

| 选项 | payload(2048 tok) | 可用性 |
|---|---|---|
| bf16 | 128 MiB | 可用 |
| **int8-group64** | **66 MiB** | **可用(推荐,显存最低)** |
| fp8 | — | sm_86 拒绝(需 Blackwell mma.f8f6f4) |
| rk8v4(q4 V) | — | 拒绝(未移植到 `kv_cache_append` Op) |

## 70K 上下文长任务测试

配置:`groupwise-int` + `--kv-dtype int8` + `--spec mtp --draft-tokens 3`,prompt 69,770 token。

| 指标 | 结果 |
|---|---:|
| prompt tokens | 69,770 |
| prefill | 636 tok/s(109.7 s) |
| decode(MTP3) | **62.97 tok/s**(91% 接受率) |
| decode(贪心 baseline) | 28.66 tok/s |
| KV payload | 2.41 GiB |
| free after startup | 625 MiB |

对比 20-token 上下文:MTP3 从 69.5 降到 62.97 tok/s,仅 ~9% 退化,注意力随上下文增长的成本被 int8 KV 有效控制。

## 推荐运行命令

```bash
./build-sm86/apps/ninfer qwen3_6_27b.ninfer \
  --messages msg.json --max-context 72000 --kv-capacity 72000 \
  --kv-dtype int8 --spec mtp --draft-tokens 3 --greedy
```
