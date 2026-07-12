ise_flow/
│
├── config/                          # 1. 配置常量与多维物理成本中心
│   ├── __init__.py                  
│   ├── schemas.py                   # 声明信息流形上的极简节点（TOKEN）类型
│   └── settings.py                  # 配置多维动态成本 alpha 向量与自适应安全门限（ζ = 0.75）
│
├── data/                            # 2. 数据编译器与 EHS 归一化模块
│   ├── __init__.py                  
│   ├── loaders.py                   # 8类数据集（GSM8K/HumanEval/FOLIO等）的本地 Parquet/JSONL 加载 [loaders.py]
│   └── parsers.py                   # 提取 CoT 步骤，利用 EHSTransformer 进行熵同态拓扑骨架去噪 [parsers.py]
│
├── llm/                             # 3. 外部推理端与自监督共形观测层
│   ├── __init__.py                  
│   ├── generator.py                 # vLLM 高并发客户端：集成 ConformalSemanticObserver 语义投影 [generator.py]
│   └── prompts.py                   # 形式化约束指令集（数学、代码、逻辑 SMT 模板） [prompts.py]
│
├── core/                            # 4. 核心不确定性量化 (UQ) 物理计算总线
│   ├── __init__.py                  
│   └── risk_estimator.py            # 终极单文件 UQ 引擎：解算双同伦共识核、迹对合密度算子与冯诺伊曼谱熵 [risk_estimator.py]
│
├── routing/                         # 5. Platt-Beta 标定、多维背包路由与对比基线计算
│   ├── __init__.py                  
│   ├── calibration.py               # 标定器：Platt-Beta 无监督自锁对齐标定（基于本征特征值自适应反馈） [calibration.py]
│   ├── router.py                    # 路由器：多维向量预算限制检查与期望损失（EVV/VCR）自适应分配 [router.py]
│   ├── verifier_manager.py          # 物理法官：托管各大任务官方测试沙箱（evalplus、Z3 求解器、SymPy） [verifier_manager.py]
│   └── baselines.py                 # 顶会对比基线库：计算 NLL, SC, SE, SAR, U_Ecc, U_Deg, P_True, LL [baselines.py]
│
├── diagnostics/                     # 6. 运行时状态审计与可视化看板监控
│   ├── __init__.py                  
│   ├── tracker.py                   # 线程安全记录样本运行时完整 UQ 决策轨迹（落盘至 JSONL） [tracker.py]
│   ├── analyzer.py                  # 审计器：解算 AUROC, AUPRC, ECE 并运行量子同伦假说交叉实证 [analyzer.py]
│   └── monitor.py                   # 看板生成器：自动将 JSONL 编译为高可视化的决策轨迹 MD 看板 [monitor.py]
│
└── run_pipeline.py                  # 7. 启动主控程序：并发调度、执行 UQ 与路由、渲染学术 LaTeX 报表 [run_pipeline.py]