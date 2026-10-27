# 论文配图：绘图思路与统一视觉规范

状态：**5 张图已按配色 A 完成**，输出在 `out/`（`.svg` 可编辑，`.pdf` 矢量用于 LaTeX，`.png` 预览）。
第一轮结构草图保留在 `out/wire/`（旧配色，仅作记录）。

| 图 | 文件 | 生成脚本 |
| --- | --- | --- |
| 图1 动态任务 + 延迟 | `out/fig1_dynamic.*` | `src/fig1_dynamic.py` |
| 图2 VLA 结构 + 10 步去噪回环 | `out/fig2_pi05.*` | `src/fig2_pi05.py` |
| 图3 异步慢/快系统四种接口 | `out/fig3_async.*` | `src/fig3_async.py` |
| 图4 Flow matching 与一步生成 | `out/fig4_flow.*` | `src/fig4_flow.py` |
| 图5 帧相机 / DVS / SNN | `out/fig5_events.*` | `src/fig5_events.py`（数据来自 `src/sim_events.py`） |

图中使用的第三方图片（请在图注中注明）：

| 图 | 内容 | 来源 |
| --- | --- | --- |
| 图1 (a)(b)(c) | 动态任务实拍 | AHEAD — Syed et al., 2026, arXiv:2606.02486, Fig. 1 |
| 图2 | 四路相机画面 | π0.5 — Physical Intelligence et al., 2025, arXiv:2504.16054, Fig. 5 |
| 图5 | 仿真背景（厨房照片） | π0.5 — arXiv:2504.16054, Fig. 6（第一张 Mock Kitchen）；运动小球、RGB 模糊帧与 DVS 事件均为仿真生成 |

图片由 `src/fetch_assets.py` 下载到 `assets/`（不入库）；把同名文件换成你们自己的照片后重新运行脚本即可替换。

## 统一视觉规范（配色 A：蓝 + 橙 + 灰）

| 项目 | 规范 |
| --- | --- |
| 画布 | 宽 1000 单位 = 7.0 in（双栏通栏），高度按内容定 |
| 字体 | 正文 Arial / Helvetica，数学变量 Times 斜体 |
| 颜色 | 每张图最多 2 个色相 + 灰：蓝 `#3D6FA5` = 慢系统 / VLM / 感知；橙 `#DD8A3E` = 快系统 / 动作；其余灰；红 `#C0392B` 只用于 1 处关键约束 |
| 事件极性 | ON = 橙，OFF = 蓝 |
| 子图 | 白底 + 细灰边框，左上角 `(a) 标题` |
| 信息粒度 | 示意图不写具体模型名 / 参数量，代表工作写在正文或图注 |
| 符号约定 | flow matching 统一 τ = 0 噪声 → τ = 1 动作：$x_τ=(1-τ)ε+τA$，目标速度 $A-ε$ |

## 构建

```bash
cd figures
python3 src/fetch_assets.py          # 下载 / 裁剪第三方图片到 assets/
python3 src/sim_events.py            # 图5 的事件相机仿真
for f in fig1_dynamic fig2_pi05 fig3_async fig4_flow fig5_events; do python3 src/$f.py; done
node src/render.js out/fig*.svg      # 生成 .pdf 与 .png
```

## 第一轮绘图思路（记录）


### 图1 动态任务示意图
- 上排三栏：(a) 传送带分拣、(b) 运动物体接取、(c) 乒乓球击打，统一侧视插画。
- 每栏下方三行：①感知（蓝）②预测与规划（紫）③执行（橙）。
- 下排 (d)：闭环流程（传感器 → 感知 → 预测与规划 → 执行 → 反馈）+ 延迟预算时间轴：目标在延迟 Δ 内移动 vΔ（例：1 m/s × 100 ms = 10 cm），所以要对预测状态行动。
- 参考：[DynamicVLA](https://arxiv.org/abs/2601.22153)、[AHEAD](https://arxiv.org/abs/2606.02486)、[ForeTime-VLA](https://www.alphaxiv.org/abs/2608.20735)。

### 图2 VLA 结构（π0.5）
- 左→右：多视角图像 / 指令 ℓ / 离散化状态 → SigLIP + Tokenizer → prefix token → VLM（PaliGemma, Gemma 2B）与 Action expert（300M）并排，逐层共享自注意力。
- VLM 上方：① 子任务 ℓ̂（自回归文本，回环拼入 prefix）；FAST token（仅预训练，虚线）。
- Action expert：噪声动作块 $A^τ$，τ 经 adaRMSNorm 注入 → $v_θ$ → 10 步 Euler → 动作块 $a_{t:t+H}$（H = 50）。
- 底部：块状注意力掩码 + 预训练 / 后训练。
- 参考：[π0.5](https://arxiv.org/abs/2504.16054)、[π0](https://arxiv.org/abs/2410.24164)，以及 GR00T N1 结构图（[Isaac-GR00T README](https://github.com/NVIDIA/Isaac-GR00T)、[LeRobot README](https://github.com/huggingface/lerobot)）的 token 竖排与迭代回环画法。
- 定稿前核对：状态离散化 bin 数、损失权重 α、注意力掩码细节（以论文与 openpi 代码为准）。

### 图3 异步 VLA 慢 / 快系统四种接口
- 顶部：共用异步时序条（慢 1–10 Hz 长条，快 50–200 Hz 短格，快系统复用最新慢输出）。
- 下方四栏同一模板：慢系统 → 接口区（紫色虚线框，只有这里各栏不同）→ 快系统 → $a_t$。
  - (a) 慢系统潜变量 → 快系统：1–2 根潜向量 z（例：Helix）。
  - (b) 中间层 Token → 快系统：第 ℓ 层引出的 token 序列，快系统 cross-attn（例：GR00T N1、RoboDual）。
  - (c) 共享网络参数：快路径复用 VLM 末几层（例：[FiS-VLA](https://arxiv.org/abs/2506.01953)）。
  - (d) 慢系统动作块 → 快系统：粗动作块 Â，快系统细化 / 跟踪（例：[Hume](https://arxiv.org/abs/2505.21432)、[RoboDual](https://arxiv.org/abs/2410.08001)）。
- 参考：FiS-VLA teaser（并排对比）、RoboDual 结构图与时间线、Hume teaser。

### 图4 Flow matching 与一步降噪动作生成
- 第一行 (a)：条件直线路径（训练）｜速度场 + 弯曲 ODE 轨迹 + Euler 点（采样）｜动作块去噪；下方统一公式条。
- (b) 蒸馏：同一噪声 $x_0$，教师 N 步弯曲轨迹 → 学生一步直线到教师终点；$L=‖S_φ(x_0,o)-Φ_T^{(N)}(x_0,o)‖^2$，NFE N→1。
- (c) 一步训练目标：瞬时速度 v（切线）vs 平均速度 $u(x_0,0,1)$（弦线），$\hat A = ε + u_θ(ε,0,1\mid o)$。
- (d) 热启动与跨步迭代：$A_{k-1}$ 平移 → 重加噪到 $τ_0$ → K = 1–2 步去噪（$o_k$）→ 执行 s 步 → 回环；红框：$‖T_o(A)-T_o(A')‖≤ρ‖A-A'‖$ **仅在观测 o 固定时成立**，$o_k$ 变化时不动点 $A^*(o_k)$ 漂移。
- 参考：[MeanFlow](https://arxiv.org/abs/2505.13447) 速度示意、[Shortcut Models](https://arxiv.org/abs/2410.12557) 方法图、[Consistency Policy](https://arxiv.org/abs/2405.07503) teaser、[Real-Time Chunking](https://arxiv.org/abs/2506.07339)。

### 图5 帧相机 / DVS 与 RGB 编码器 / SNN 对比
- 三行管线 × 五列（传感器 / 原始数据 / 编码器输入 / 计算方式 / 特性）。
- (a) 帧相机 + RGB 编码器：33 ms 一帧、运动模糊 → 稠密 MAC。
- (b) DVS + RGB（ANN）编码器：x–y–t 事件点云 → 时间窗体素化 B×H×W（丢失稀疏性与时间精度）→ 稠密 MAC。
- (c) DVS + SNN：脉冲序列 → LIF 积分 / 发放 / 复位 → 仅在脉冲到达时做 AC。
- 可选脚注：$E_{MAC}≈4.6$ pJ，$E_{AC}≈0.9$ pJ（32-bit FP，45 nm，Horowitz 2014）。
- 参考：[event_utils](https://github.com/TimoStoff/event_utils) 事件表示图、[Gallego et al. 事件相机综述](https://arxiv.org/abs/1904.08405)。
