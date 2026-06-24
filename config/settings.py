# -*- coding: utf-8 -*-
"""ISE-Flow 实验配置与超参数管理.

该模块集成了第一阶段（ISE-Router）和长期自进化阶段（ER-EvoGraph + E-CIG）的
所有超参数、多维动态成本参数、分桶分量配置和全局门控阈值。
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class GraphCoarseningConfig:
    """多尺度图粗粒化（HEM/Graclus）配置参数."""
    max_scale_K: int = 5           # HEM 局部聚合的最大尺度深度（K） [PDF Page 3]
    min_node_threshold: int = 3    # 触发聚合终止的最小节点数上限
    edge_affinity_quantile: float = 0.7  # 粗粒化 affinities 合并停止的无监督分位数阈值


@dataclass(frozen=True)
class NullCacheConfig:
    """离线 Null-cache 标准化检索配置."""
    num_quantile_bins: int = 10    # |V|, |E|, density, type_entropy 的等频分桶数 [PDF Page 8]
    knn_k_default: int = 10        # Standardized Euclidean KNN 默认近邻数（k） [PDF Page 8]
    knn_k_sparse_fallback: int = 5 # 缓存极端稀疏时的降级近邻数 [PDF Page 8]
    distance_threshold_D_max: float = 2.5  # 判定为 Near-hit 的最大标准化欧氏距离阈值


@dataclass(frozen=True)
class RISEWeightsConfig:
    """ISE 综合结构风险分数（R_ISE）融合权重.
    
    采用 Fixed Rank Aggregation 机制，用于无监督开发集。
    公式对应：R_ISE(τ) = ws * R_struct + wp * R_parse + wr * R_robust + wc * R_cache
    """
    w_struct: float = 0.45         # ws: 多尺度结构风险权重 [PDF Page 4]
    w_parse: float = 0.25          # wp: 解析异常与覆盖度风险权重 [PDF Page 4]
    w_robust: float = 0.20         # wr: 规则格式扰动稳定性风险权重 [PDF Page 4]
    w_cache: float = 0.10          # wc: Cache Near-hit 偏离度及 Miss 惩罚权重 [PDF Page 4]

    def __post_init__(self):
        """校验权重和在数值上的合理性（非强制归一，但建议进行正数校对）."""
        if any(w < 0 for w in (self.w_struct, self.w_parse, self.w_robust, self.w_cache)):
            raise ValueError("All RISE weights must be non-negative.")


@dataclass(frozen=True)
class CostBudgetConfig:
    """多维动态成本与批处理验证预算配置.
    
    标量化公式：C(v) = α^T * c(v)
    """
    # 动态成本维度权重 α: ($成本, 延迟, 计算资源, 物理风险, 隐私泄露风险, 状态冲突)
    alpha_vector: Tuple[float, ...] = (0.4, 0.2, 0.1, 0.1, 0.1, 0.1)

    # 验证器多维综合预算上限 B: ($预算, 延迟限值, 算力配额, 最大承受风险, 隐私安全限值) [PDF Page 4]
    budget_vector_20: Tuple[float, ...] = (100.0, 500.0, 1000.0, 10.0, 5.0)  # 20% 档位预算 [PDF Page 8]
    budget_vector_30: Tuple[float, ...] = (150.0, 750.0, 1500.0, 15.0, 7.5)  # 30% 档位预算 [PDF Page 8]

    # 前置计算成本过线阀值比率
    # 约束：C_pre <= pre_routing_ratio * C_cheap_verifier (典型要求 <= 0.3) [PDF Page 8]
    pre_routing_ratio: float = 0.3 
    pre_routing_target: float = 0.2 # 强结果期望目标 <= 0.2 [PDF Page 8]


@dataclass(frozen=True)
class ECIGGateConfig:
    """长期阶段 E-CIG 受控编辑决策树配置 (ER-EvoGraph).
    
    约束公式：I(Y; Δ_t) - λD * D_struct - λC * C_multi - λR * R_reg - λK * R_compat - λO * [τ_O - O_t]_+ > 0
    """
    lambda_D_struct: float = 0.5   # λD: 结构熵漂移（信息污染）惩罚权重 [PDF Page 8]
    lambda_C_multi: float = 0.5    # λC: 多维持久维护与状态编辑成本惩罚权重 [PDF Page 8]
    lambda_R_reg: float = 1.0      # λR: 回归测试失败率（通过率退化）惩罚权重 [PDF Page 8]
    lambda_K_compat: float = 0.5   # λK: Harness/Policy 兼容性不匹配风险惩罚权重 [PDF Page 8]
    lambda_O_observability: float = 1.0 # λO: 反馈缺失/不可观测性惩罚权重 [PDF Page 8]
    
    observability_threshold_tau_O: float = 1.0 # τ_O: 触发静默、隔离或拒写入的反馈观测阈值 [PDF Page 8]


@dataclass(frozen=True)
class SafetyGateConfig:
    """解析稳定性与语法校验阈值."""
    coverage_threshold_zeta: float = 0.75  # ζ: Parser 覆盖率安全门限 [PDF Page 8]


@dataclass
class ExperimentSettings:
    """全局实验配置类（聚合所有子配置项）."""
    coarsening: GraphCoarseningConfig = field(default_factory=GraphCoarseningConfig)
    cache: NullCacheConfig = field(default_factory=NullCacheConfig)
    rise_weights: RISEWeightsConfig = field(default_factory=RISEWeightsConfig)
    cost_budget: CostBudgetConfig = field(default_factory=CostBudgetConfig)
    ecig_gate: ECIGGateConfig = field(default_factory=ECIGGateConfig)
    safety_gate: SafetyGateConfig = field(default_factory=SafetyGateConfig)

    # 运行基本配置
    random_seed: int = 2048    # 保持严格学术复现的固定全局种子
    use_calibration: bool = True    # 测试阶段是否默认启用分桶概率校准
    
    def to_dict(self) -> Dict[str, any]:
        """将当前实验的所有静态配置转化为字典，用于诊断模块 tracker 的序列化输出."""
        return asdict(self)

    def to_json(self) -> str:
        """将当前实验配置转化为标准格式化的 JSON 字符串."""
        return json.dumps(self.to_dict(), indent=4, ensure_ascii=False)