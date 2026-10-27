# 论文配图：绘图思路与统一视觉规范

状态：**结构草图阶段，等待确认风格与结构后再完整绘制。**
结构草图在 `out/wire/`，生成脚本在 `src/`。

## 统一视觉规范（5 张图共用）

| 项目 | 规范 |
| --- | --- |
| 画布 | 宽 1000 单位 = 7.0 in（双栏通栏），高度按内容定 |
| 字体 | 正文 Arial / Helvetica，数学变量 Times 斜体 |
| 字号 | 子图标题 9 pt 粗体；模块名 8 pt 粗体；次要文字 7 pt 灰；注释最小 6.3 pt |
| 模块 | 浅色填充 + 同色深描边，圆角 6–10 |
| 子图背景 | 浅灰圆角底板 `#F6F7F9`，左上角 `(a) 标题` |
| 箭头 | 实线：数据流；虚线：反馈 / 可选；粗紫线：慢→快接口；橙色点线：规划轨迹 |
| 插画 | 扁平侧视，机械臂为灰色胶囊连杆 + 深色关节，全文同一造型 |
| 符号约定 | flow matching 统一用 τ = 0 为噪声、τ = 1 为动作：$x_τ=(1-τ)ε+τA$，目标速度 $A-ε$ |

语义配色（贯穿全文，同一概念同一颜色）：

| 颜色 | 描边 / 填充 | 含义 |
| --- | --- | --- |
| 蓝 | `#2F6DB5` / `#E4EEF9` | 感知、视觉、图像 token |
| 紫 | `#6C4FB8` / `#EEE9F8` | 慢系统、VLM、规划 |
| 橙 | `#D9772B` / `#FCEEDF` | 快系统、动作专家、执行 |
| 绿 | `#3B9A5F` / `#E4F3E9` | 语言 / 状态 token |
| 青 | `#178A87` / `#DFF2F1` | 脉冲 / 神经形态（SNN） |
| 灰 | `#8A919E` / `#EEF0F3` | 噪声、中性元素 |
| 红 | `#C8423B` | 关键约束 / 警示（如“收缩仅在观测固定时成立”） |
| 红 / 蓝点 | `#D6453D` / `#2F5FC4` | 事件极性 ON / OFF |

## 各图绘图思路

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

## 构建

```bash
cd figures
python3 src/styleguide.py && python3 src/wireframes.py
node src/render.js out/wire/*.svg     # 输出同名 .pdf（矢量，7 in 宽）与 .png（预览）
```
