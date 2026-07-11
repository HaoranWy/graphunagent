ise_flow/
│
├── config/                          # 全局实验常数与多维物理成本配置
│   ├── __init__.py                  
│   ├── schemas.py                   # 定义极简的 TOKEN 节点与 ATTENTION 边拓扑常数
│   └── settings.py                  # 配置多维物理成本向量与全局安全门控（如解析覆盖率阈值 ζ）
│
├── data/                            # 数据编译器与格式化 Ingestion 模块
│   ├── __init__.py                  
│   ├── loaders.py                   # 负责 GSM8K, HumanEvalPlus, FOLIO 等 8 类数据集的安全加载
│   └── parsers.py                   # 提取 CoT 步骤与 EHS 骨架化去噪归一化器（EHSNormalizer）
│
├── core/                            # 核心不确定性量化 (UQ) 估算中心
│   ├── __init__.py                  
│   └── risk_estimator.py            # 主引擎：实现自监督双同伦谱投影、量子混态密度算子与冯诺伊曼熵解算
│
├── llm/                             # 外部推理端与自监督观测层
│   ├── __init__.py                  
│   ├── generator.py                 # vLLM 高并发客户端：集成子词共形投影观测器（LSO）
│   └── prompts.py                   # 形式化约束指令集（数学、代码、逻辑 SMT 模板）
│
├── routing/                         # Platt-Beta 标定、多维背包路由与对比基线计算
│   ├── __init__.py                  
│   ├── calibration.py               # Platt-Beta 无监督极性自锁对齐标定器（基于 MSO 在线辨识）
│   ├── router.py                    # 路由器：多维物理成本剩余预算 B 边界检查与期望损失最小化兜底分配
│   ├── verifier_manager.py          # 物理法官：托管 8 类数据集官方评估引擎（如 evalplus 物理沙箱、SymPy、Z3）
│   └── baselines.py                 # 顶会基线库：计算 NLL, SC, Self-Certainty, SE, SAR, U_Ecc, U_Deg
│
├── diagnostics/                     # 运行时状态审计与可视化监控
│   ├── __init__.py                  
│   ├── tracker.py                   # 线程安全记录样本决策轨迹指标（落盘至 JSONL 诊断日志）
│   ├── analyzer.py                  # 审计器：解算 AUROC, AUPRC, ECE 并运行不变量显著性检验
│   └── monitor.py                   # 自动将 JSONL 日志编译为全流程决策轨迹 HTML/Markdown 看板
│
└── run_pipeline.py                  # 启动主控程序：调度并发采样、执行 UQ 与路由、渲染学术 LaTeX 报表