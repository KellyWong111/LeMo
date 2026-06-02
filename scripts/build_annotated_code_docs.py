from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "annotated_code"


FILES = [
    (
        "code/moda_module_exact.py",
        "MoDA exact-kernel predictor",
        [
            "这是最核心的 MoDA 结构接入代码之一。",
            "它尽量保留 LeWM predictor 原始结构，只把 predictor attention 替换成 MoDA Triton kernel。",
            "汇报时重点讲：cached_k/cached_v 表示跨层 depth cache；有 cache 时走 MoDA，多深度检索；无 cache 时退回普通 causal attention。",
        ],
    ),
    (
        "code/moda_module.py",
        "MoDA fallback / non-exact module",
        [
            "这是另一个 MoDA 模块实现，通常用于非 exact-kernel 或较早版本实验。",
            "汇报时重点讲：它体现了把 depth attention 融入 LeWM predictor 的工程路径。",
        ],
    ),
    (
        "code/train_encoder_moda.py",
        "MoDA encoder training entry",
        [
            "这是把 MoDA 接入 LeWM encoder/predictor 训练的入口。",
            "汇报时重点讲：这一步属于结构接入阶段，目标是让 MoDA representation 能进入 LeWM planning pipeline。",
        ],
    ),
    (
        "code/train_moda.py",
        "MoDA training script",
        [
            "这是 MoDA 早期训练入口。",
            "汇报时重点讲：它对应最早验证 MoDA module 是否能训练、是否能替换原 attention/predictor 的阶段。",
        ],
    ),
    (
        "code/train_moda_exact.py",
        "MoDA exact-kernel training script",
        [
            "这是 exact MoDA kernel 路径的训练入口。",
            "汇报时重点讲：它对应后面更接近 official MoDA Triton kernel 的强实现。",
        ],
    ),
    (
        "experiments/moda_only_intra_episode_audit.py",
        "MoDA-only intra-episode audit",
        [
            "这是证明 AUC 假阳性的关键诊断脚本。",
            "它区分 global AUC 和 intra-episode AUC，解释为什么 BCE/AUC 提升不等于 top1 planning 提升。",
        ],
    ),
    (
        "experiments/moda_only_learned_residual_proposal.py",
        "MoDA-only learned residual proposal",
        [
            "这是当前最重要的 MoDA-only 正向方向。",
            "它不是 final rerank，而是学习 delta_a = success_action - raw_rank0_failure_action，修正 action proposal generation。",
            "汇报时重点讲 train_data、fit_ridge、online_feature、score_candidates、run_split 这几段。",
        ],
    ),
    (
        "experiments/moda_only_residual_confirm50_audit.py",
        "Residual proposal paired confirm50 audit",
        [
            "这是最后确认 residual proposal 是否稳定的 paired-index 审计脚本。",
            "它保存/复用 eval indices，并拆 first20 / added30 / all50，避免 medium20 和 confirm50 采样集合不同导致误判。",
        ],
    ),
    (
        "experiments/moda_only_planner_in_loop_calibrated_cem.py",
        "Planner-in-the-loop calibrated CEM",
        [
            "这是尝试把 calibrated utility 注入 CEM 内部的 MoDA-only 路线。",
            "它的意义是诊断：只在 CEM 中改 cost 有小幅提升，但不足以形成稳定 standalone 方法。",
        ],
    ),
    (
        "experiments/moda_only_action_sensitive_contrastive.py",
        "Action-sensitive contrastive diagnostic",
        [
            "这是尝试让 success candidate 和 near-miss failure 在 embedding 上分开的诊断脚本。",
            "最终 top1 不好，所以作为负结果和机制诊断，而不是主方法。",
        ],
    ),
    (
        "experiments/risk_controlled_moda_integration.py",
        "Risk-controlled baseline-safe MoDA integration",
        [
            "这是 bsl-relative integration 线的代表。",
            "它能提升系统 top1，但依赖 baseline fallback，所以不能当作 MoDA-only 主结果。",
        ],
    ),
    (
        "experiments/pac_moda_native_calibration_report.py",
        "PAC-MoDA native calibration report",
        [
            "这是把 MoDA-native calibration 和 localized activation 分开的报告脚本。",
            "汇报时重点讲：global ranking 表只看 raw/BCE/rank-preserve；localized activation 不能混进 global ranking。",
        ],
    ),
]


KEY_TERMS = {
    "ROOT": "实验输出根目录，通常指向 /data1/jingyixi/wm_runs。",
    "ST_ACTION": "stateroll/MoDA candidate pool 中的 proposal/action/cost/label 数据目录。",
    "ST_RAW": "stateroll/MoDA candidate pool 中的 raw rollout npz 目录，包含预测轨迹等。",
    "OUT": "当前脚本的输出目录。",
    "POLICY": "要加载的 LeWM/MoDA policy checkpoint 名称。",
    "SPLITS": "训练/验证 seed split，用于交叉验证或 paired audit。",
    "goal_for_pred": "把 goal tensor 调整到和 pred rollout 同样的维度，方便计算距离。",
    "pool_stats": "从候选轨迹中抽取 final/mean/min distance、progress、latent/action 统计特征。",
    "feature_np": "构造离线候选特征，包含 raw cost、rank、z-score、trajectory/action 特征。",
    "train_data": "从候选池里构造监督训练样本。",
    "fit_ridge": "用 ridge regression 学 residual 或 utility 的线性权重。",
    "predict_ridge": "用训练好的 ridge 模型预测分数或 residual。",
    "online_feature": "在线 CEM 评估时重新构造和离线训练一致的特征。",
    "score_candidates": "对候选 action rollout，计算 raw cost、calibrated cost 和 utility。",
    "labels_to_row": "把每个候选的 success label 汇总成 top1/top3/top5/oracle/near-miss 指标。",
    "run_split": "单个 train/val split 的主实验流程。",
    "choose_indices": "固定随机种子抽 eval indices，避免不同 run 的样本集合混淆。",
    "paired_metrics": "计算相对 raw MoDA 的 fixed/harmed/net，用于 paired audit。",
    "MoDAAttentionExact": "核心 attention 替换模块；有 depth cache 时调用 MoDA kernel。",
    "MoDAConditionalBlockExact": "带 AdaLN-zero conditioning 的 LeWM block，把 attention 换成 MoDA attention。",
    "MoDATransformerExact": "多层 MoDA conditional blocks 组成的 predictor 主体。",
    "_build_depth_cache": "把前面层的 K/V stack 成 depth cache，供 MoDA 跨层检索。",
}


def explain_line(stripped: str) -> str:
    if not stripped:
        return ""
    if stripped.startswith('"""') or stripped.startswith("'''"):
        return "文档字符串：说明这个文件/函数/类的用途。"
    if stripped.startswith("from __future__"):
        return "启用新版 Python annotation 行为，减少类型注解运行时副作用。"
    if stripped.startswith("import ") or stripped.startswith("from "):
        return "导入依赖模块。"
    if stripped.startswith("os.environ.setdefault"):
        return "设置默认环境变量，保证 headless MuJoCo / EGL 环境能运行。"
    if stripped.startswith("class "):
        name = stripped.split("(")[0].replace("class ", "").replace(":", "")
        return f"定义类 `{name}`，这是该文件的一个主要模块。"
    if stripped.startswith("def "):
        name = stripped.split("(")[0].replace("def ", "")
        return f"定义函数 `{name}`，封装一段可复用逻辑。"
    if stripped.startswith("@"):
        return "装饰器：改变下面函数的运行方式，例如关闭梯度或标注类型。"
    if stripped.startswith("if __name__"):
        return "脚本入口：直接运行该文件时从 main() 开始。"
    if stripped.startswith("if "):
        return "条件分支：根据当前状态选择不同逻辑。"
    if stripped.startswith("elif "):
        return "条件分支的其他情况。"
    if stripped.startswith("else"):
        return "条件分支的兜底情况。"
    if stripped.startswith("for "):
        return "循环：遍历多个 seed、candidate、layer 或配置。"
    if stripped.startswith("while "):
        return "循环：持续执行直到条件不满足。"
    if stripped.startswith("try"):
        return "异常保护：尝试导入或执行可能失败的逻辑。"
    if stripped.startswith("except"):
        return "异常处理：导入失败或运行失败时走 fallback。"
    if stripped.startswith("return "):
        return "返回当前函数的结果。"
    if stripped.startswith("raise "):
        return "显式报错，避免无效输入继续运行。"
    if stripped.startswith("parser.add_argument") or stripped.startswith("ap.add_argument"):
        return "命令行参数：控制实验规模、输出目录或超参数。"
    if "np.load" in stripped:
        return "读取离线候选池 / rollout / label 数据。"
    if "torch.no_grad" in stripped or "torch.inference_mode" in stripped:
        return "推理阶段关闭梯度，节省显存并避免污染模型参数。"
    if "argmin" in stripped and "cost" in stripped.lower():
        return "按 raw cost 找当前 MoDA 认为最优的候选，通常就是 rank0。"
    if "labels" in stripped and "success" in stripped:
        return "使用 success label 判断候选是否真实规划成功。"
    if "near_miss" in stripped:
        return "near-miss 指标：top1 失败但候选池里存在成功候选。"
    if "top1" in stripped or "top3" in stripped or "top5" in stripped or "oracle" in stripped:
        return "规划评估指标：top1 是最终选择成功率，oracle 是候选池内是否至少有成功。"
    if "delta" in stripped and ("success" in stripped or "rank0" in stripped or "residual" in stripped):
        return "学习从失败 action 到成功 action 的 residual，这是 residual proposal 的核心。"
    if "torch.cat" in stripped or "np.concatenate" in stripped:
        return "拼接特征或候选集合。"
    if "torch.stack" in stripped or "np.stack" in stripped:
        return "堆叠多个张量/数组形成 batch。"
    if "args." in stripped:
        return "读取命令行参数，保证实验配置可复现。"
    if "=" in stripped and not stripped.startswith(("==", ">=", "<=")):
        left = stripped.split("=", 1)[0].strip()
        if left in KEY_TERMS:
            return KEY_TERMS[left]
        return "变量赋值：保存中间结果或配置。"
    return "执行当前逻辑的一行代码。"


def write_annotated(src_rel: str, title: str, summary: list[str]) -> None:
    src = ROOT / src_rel
    if not src.exists():
        return
    text = src.read_text(errors="replace").splitlines()
    out_name = src_rel.replace("/", "__").replace(".py", ".annotated.md")
    out = OUT / out_name
    lines = [
        f"# {title}",
        "",
        f"Original file: `{src_rel}`",
        "",
        "## 汇报用总览",
        "",
    ]
    lines.extend(f"- {item}" for item in summary)
    lines.extend(
        [
            "",
            "## 关键术语",
            "",
        ]
    )
    for key, value in KEY_TERMS.items():
        if any(key in line for line in text):
            lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## 逐行/分段注释",
            "",
            "> 格式说明：每一行先给原始代码，再给中文解释。注释版用于汇报，不替代可运行源码。",
            "",
        ]
    )
    for idx, raw in enumerate(text, start=1):
        stripped = raw.strip()
        explanation = explain_line(stripped)
        lines.append(f"### L{idx}")
        lines.append("")
        lines.append("```python")
        lines.append(raw)
        lines.append("```")
        if explanation:
            lines.append(f"说明：{explanation}")
        else:
            lines.append("说明：空行，用于分隔代码结构。")
        lines.append("")
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index = [
        "# Annotated Code Index",
        "",
        "这些文件是汇报用中文注释版，原始可运行代码仍在 `code/` 和 `experiments/`。",
        "",
    ]
    for rel, title, summary in FILES:
        write_annotated(rel, title, summary)
        out_name = rel.replace("/", "__").replace(".py", ".annotated.md")
        index.append(f"- [{title}]({out_name}) - `{rel}`")
    (OUT / "README.md").write_text("\n".join(index) + "\n")


if __name__ == "__main__":
    main()
