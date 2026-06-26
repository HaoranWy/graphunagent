ise_flow/
├── config/
│   ├── schemas.py           # 拓扑约束器：严格规范异构有向图节点 (STEP, CLAIM, ANSWER...) 与边 (SEQUENCE, SUPPORT...) 的合法连接Schema
│   └── settings.py          # 参数配置总线：托管多维成本向量权重、HEM 粗粒化、Null-cache 分桶及物理预算限制
│
├── data/
│   ├── loaders.py           # 数据加载器：支持 GSM8K（Parquet）与 HumanEvalPlus（Parquet）测试集物理读取
│   ├── parsers.py           # 归一化编译器：集成 EHS-Normalizer 骨架收缩器，完成变量作用域重命名与线性控制流折叠
│   └── graph_builder.py     # 拓扑组装器：实现无监督二分符号投影图、高维 Python AST 代码图的构建
│
├── core/
│   ├── stcg.py              # 谱共识引擎：重构为“层论多路径同伦相容性拼缝”算法，支持多路径算法多样性（STCG）
│   ├── coarsening.py        # HEM 粗粒化器：执行带有词典序有序最大独立集匹配与超级节点属性合并
│   ├── null_cache.py        # 零模型检索引擎：利用 MCMC PCFG-AST 树形产生式网格填充与全局 KNN-IDW 回归
│   └── risk_estimator.py    # 拓扑风险编译引擎（核心）：实现“纯代数、参数无关的捷赫子空间投影层（CSS-CCR）”，自适应静态与动态测试流复形（DEC）
│
├── routing/
│   ├── calibration.py       # 标定器：非参数分桶校准器、KDE 弱监督噪声矩阵自抗扰标定器（RWSC）
│   ├── verifier_manager.py  # 验证器沙箱：托管线程隔离的代码执行环境，注入了 typing/collections 等高阶库规避沙箱崩溃
│   └── router.py            # 路由器：多维预算向量约束下，期望效用最大化（VCR）背包分配
│
├── diagnostics/
│   ├── tracker.py           # 运行时日志：原子化追加物理轨迹与分尺度高维度捷赫阻碍特征流至本地 JSONL
│   ├── analyzer.py          # 指标审计：自适应统计 EDR, SSO, LDI, ECE, Brier Score 以及分层 AUROC 核心质量指标
│   └── monitor.py           # 看板：编译生成端到端决策链追踪大面板
│
├── experiments/
│   └── run_exp.py           # 实验调度与编排总线：协调 Dev 标定拟合、Test 测试路由及 Tracker 落地全闭环
│
└── run_pipeline.py          # 启动主程序：自动离线数据集环境绑定，支持 evalplus 子进程多 Key 兼容解析与自适应降级