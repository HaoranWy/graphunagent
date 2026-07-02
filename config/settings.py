# -*- coding: utf-8 -*-
"""ISE-Flow 实验配置与多维物理成本管理.

该模块已彻底剥离图粗粒化与缓存常数，专注于多维动态成本参数与全局门控。
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class RISEWeightsConfig:
    """ISE 综合结构风险分数（R_ISE）融合权重.
    
    用于融合单路径路径曲率风险、解析异常和扰动异常。
    """
    w_struct: float = 0.60         # 路径/局部投影曲率风险权重
    w_parse: float = 0.20          # 解析异常与覆盖度风险权重
    w_robust: float = 0.20         # 规则格式扰动稳定性风险权重


@dataclass(frozen=True)
class CostBudgetConfig:
    """多维动态成本与批处理验证预算配置.
    
    标量化公式：C(v) = α^T * c(v)
    """
    # 动态成本维度权重 α: ($成本, 延迟, 计算资源, 物理风险, 隐私泄露风险, 状态冲突)
    alpha_vector: Tuple[float, ...] = (0.4, 0.2, 0.1, 0.1, 0.1, 0.1)

    # 验证器多维综合预算上限 B: ($预算, 延迟限值, 算力配额, 最大承受风险, 隐私安全限值)
    budget_vector_20: Tuple[float, ...] = (100.0, 500.0, 1000.0, 10.0, 5.0)  # 20% 档位预算
    budget_vector_30: Tuple[float, ...] = (150.0, 750.0, 1500.0, 15.0, 7.5)  # 30% 档位预算

    # 前置计算成本过线阀值比率
    pre_routing_ratio: float = 0.3 
    pre_routing_target: float = 0.2 


@dataclass(frozen=True)
class SafetyGateConfig:
    """解析稳定性与语法校验阈值."""
    coverage_threshold_zeta: float = 0.75  # ζ: Parser 覆盖率安全门限


@dataclass
class ExperimentSettings:
    """全局实验配置类（聚合所有子配置项）."""
    rise_weights: RISEWeightsConfig = field(default_factory=RISEWeightsConfig)
    cost_budget: CostBudgetConfig = field(default_factory=CostBudgetConfig)
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