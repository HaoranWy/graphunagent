ise_flow/
├── config/
│   ├── schemas.py           # 拓扑常数声明：仅定义极简的 TOKEN 节点与 ATTENTION 边符号。
│   └── settings.py          # 配置总线：彻底剥离 coarsening 与 cache 配置，管理多维物理成本预算。
│
├── data/
│   ├── loaders.py           # 数据加载器：负责 GSM8K 与 HumanEval+ 数据集的本地 Parquet 物理读取。
│   └── parsers.py           # 归一化编译器：集成 EHS-Normalizer 骨架化去噪器与数学答案 boxed 提纯。
│
├── core/
│   ├── risk_estimator.py    # 信息几何风险评估引擎：实现 IJP 解析、LDAS 自适应自缩放谱隙求解、EPA 商空间同伦对齐与假设指标随动回传。[核心重构]
│   └── stcg.py              # 谱共识引擎：多路径同伦谱共识特征对齐。
│
├── llm/
│   ├── generator.py         # 并发生成端：支持高并发 vLLM 调用，原生配置 top_logprobs 导出单纯形本征分布。[核心重构]
│   └── prompts.py           # 轨迹 Prompt 模板：定义高可解析性格式约束指令。
│
├── routing/
│   ├── calibration.py       # 变分标定器：RVSD 秩次极性对齐一维二分凸收敛寻根器，集成基于 MSO 均值的 OOD 自适应动态标定。[核心重构]
│   ├── verifier_manager.py  # 验证器管理器：托管线程隔离执行沙箱（GSM8K 官方标准提纯、HumanEval+）。
│   ├── router.py            # 成本路由器：执行多维资源预算边界检查与期望验证价值（EVV）背包路由。
│   └── baselines.py         # 基线计算器：实现经过 CoT 去噪调整后的 NLL、SC、DeepConf 等对比算法。
│
├── diagnostics/
│   ├── tracker.py           # 指标追踪器：原子化记录变分状态。
│   ├── analyzer.py          # 指标审计器：实现在线计算 ECE、Brier Score 及同分平局鲁棒（Fractional Ranking）的标准 AUROC。
│   └── monitor.py           # 看板生成器：编译生成决策链追踪仪表盘。
│
└── run_pipeline.py          # 启动主程序：整合单/多路径独立标定，集成本征信息几何诊断台与假设指标实证分析。[核心重构]