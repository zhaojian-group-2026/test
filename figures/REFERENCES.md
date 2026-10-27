# 参考图来源（第二轮）

图注里标注引用时，用“论文 + 图号”即可。

## 图1 动态任务 + 延迟

| 参考图 | 来源 | 借鉴点 |
| --- | --- | --- |
| Fig. 1 | Chronos Policy — Wang et al., 2026, [arXiv:2610.13929](https://arxiv.org/abs/2610.13929) | 多传感器时间轴 t_obs → t_input → t_output → t_act，编号标注各类延迟（延迟部分主参考） |
| Fig. 1(a)(b) | DynamicVLA — Xie et al., 2026, [arXiv:2601.22153](https://arxiv.org/abs/2601.22153) | 观测编号与动作块对齐，标出感知–执行间隙（P.E. Gap） |
| Fig. 1 | AHEAD — Syed et al., 2026, [arXiv:2606.02486](https://arxiv.org/abs/2606.02486) | 动态任务实拍 + 半透明轨迹叠加 |

## 图2 VLA 结构（π0.5）+ 10 步 flow matching 回环

| 参考图 | 来源 | 借鉴点 |
| --- | --- | --- |
| Fig. 3 | π0.5 — Physical Intelligence et al., 2025, [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) | 官方模型总览（预训练 / 后训练与推理），主参考 |
| Fig. 18 | π0.5 — 同上 | 注意力掩码（action expert 不看 FAST token） |
| Fig. 3 | π0 — Black et al., 2024, [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) | action expert 内虚线回环表示迭代去噪 |
| Fig. 1 | Knowledge Insulation — Driess et al., 2025, [arXiv:2505.23705](https://arxiv.org/abs/2505.23705) | 损失图例 + stop-gradient |
| Fig. 1（下半） | SmolVLA — Shukor et al., 2025, [arXiv:2506.01844](https://arxiv.org/abs/2506.01844) | 输出绕回 noisy actions 的虚线大回环 |

## 图3 异步慢 / 快系统四种接口

| 参考图 | 来源 | 对应 |
| --- | --- | --- |
| Fig. 1 | DuoCore-FS — Zou et al., 2025, [arXiv:2512.20188](https://arxiv.org/abs/2512.20188) | Bridge Buffer：慢系统 1–3 Hz 写入 hidden-state token，快系统 25–30 Hz 读取；接口 (b) + 异步 |
| Fig. 1 | UniFS — Sun et al., 2026, [arXiv:2606.22794](https://arxiv.org/abs/2606.22794) | 多种范式并排对比，四栏版式 |
| Fig. 4 | Efficient VLA Survey — Guan et al., 2025, [arXiv:2510.17111](https://arxiv.org/abs/2510.17111) | latent vector；“每 N 步推理 / 每步执行”；接口 (a) |
| Fig. 1 / Fig. 2 | FiS-VLA — Chen et al., 2025, [arXiv:2506.01953](https://arxiv.org/abs/2506.01953) | System 1 = VLM 末几层；接口 (c) |
| Fig. 2 | Libra-VLA — Wei et al., 2026, [arXiv:2604.24921](https://arxiv.org/abs/2604.24921) | 粗动作 → Intent Buffer → 细化；接口 (d) |
| Fig. 2 | Hume — Song et al., 2025, [arXiv:2505.21432](https://arxiv.org/abs/2505.21432) | 候选动作块 + 级联去噪；接口 (d) |
| Fig. 2 | SmolVLA — Shukor et al., 2025, [arXiv:2506.01844](https://arxiv.org/abs/2506.01844) | 异步推理时间线 |
| Fig. 1 | HiRT — Zhang et al., 2024, [arXiv:2410.05273](https://arxiv.org/abs/2410.05273) | 低频 VLM / 高频策略（可选） |

## 图4 Flow matching 与一步降噪（结构已确认）

| 参考图 | 来源 | 对应 |
| --- | --- | --- |
| Fig. 2 | MeanFlow — Geng et al., 2025, [arXiv:2505.13447](https://arxiv.org/abs/2505.13447) | 条件速度 vs 边缘速度场（第一行原理） |
| Fig. 3 | MeanFlow — 同上 | 瞬时速度 vs 平均速度（栏 c） |
| Fig. 3 | Shortcut Models — Frans et al., 2024, [arXiv:2410.12557](https://arxiv.org/abs/2410.12557) | 多步轨迹 vs 大步长跳跃（栏 b） |
| Fig. 1 | Consistency Policy — Prasad et al., 2024, [arXiv:2405.07503](https://arxiv.org/abs/2405.07503) | 机械臂场景的多步 vs 一步（栏 b） |
| （文字引用） | Real-Time Chunking — Black et al., 2025, [arXiv:2506.07339](https://arxiv.org/abs/2506.07339) | 上一动作块热启动 / inpainting（栏 d） |

## 图5 帧相机 / DVS + RGB 编码器 / DVS + SNN

| 参考图 | 来源 | 对应 |
| --- | --- | --- |
| Fig. 1 | Event-based Vision Survey — Gallego et al., TPAMI 2020, [arXiv:1904.08405](https://arxiv.org/abs/1904.08405) | DVS 原理、事件螺旋、帧 + 事件叠加（传感器侧） |
| README 示意图 | [event_utils](https://github.com/TimoStoff/event_utils)（T. Stoffregen） | x–y–t 事件点云 → 体素网格 → 事件帧；引用前请核对它最早出自哪篇论文 |
| Fig. 3 | E-VLA — Zhai et al., 2026, [arXiv:2604.04834](https://arxiv.org/abs/2604.04834) | 帧与事件共用时间轴，事件累计 → 表示 → 与图像编码器融合（第二行） |
| Fig. 2 | E-VLA — 同上 | Overlay 与分层 Event Adapter 两种融合方式 |
| Fig. 3 | ES-Parkour — Zhang et al., 2025, [arXiv:2503.09985](https://arxiv.org/abs/2503.09985) | 事件点云 → SNN Encoder；ANN 神经元 vs 脉冲神经元图例（第三行） |
| Fig. 2 | Abdelrahman et al., 2024, [arXiv:2404.05858](https://arxiv.org/abs/2404.05858) | ANN 全激活 vs SNN 稀疏发放（突出纯事件驱动） |
| Fig. 4 | 同上 | Event Data → Convolutional SNN → Spiking Activity 流水线 |
