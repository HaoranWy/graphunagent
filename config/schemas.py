# -*- coding: utf-8 -*-
"""ISE-Flow 轨迹图拓扑 Schema 规范.

该模块定义了推理/工具轨迹转化为异构有向图（Heterogeneous Directed Graph）时所需的
所有节点类型、边类型、图元素实体以及异构约束Schema（Heterogeneous Constraints Schema）。

对应理论公式：
    - L1/L2 轨迹图: G_0 = (V, E, w, r, p)
    - 长期持久图 (ER-EvoGraph): G_t = (VT ∪ VH ∪ VE ∪ VM ∪ VP, E)
"""

from enum import Enum, unique
from typing import Dict, Set, Tuple, Any, Optional, List
from dataclasses import dataclass, field


@unique
class NodeType(Enum):
    """ISE-Flow 异构有向图中的节点类型（NodeType）.
    
    包含 L1 推理步骤节点（VT）、形式化工件节点（VF）、L2 工具调用与执行节点（VA）
    以及 ER-EvoGraph 持久自进化状态节点（VH, VE, VM, VP）。
    """
    # =====================================================================
    # L1 推理与事件轨迹层 (VT - Trajectory / Event Level Nodes)
    # =====================================================================
    STEP = "STEP"             # 链式推理步骤或自然语言推理行（Step Chain）
    CLAIM = "CLAIM"            # 推理中间断言、命题或子结论
    ANSWER = "ANSWER"          # 最终提交的答案或推理终止结论

    # =====================================================================
    # L1 形式化/AST 层 (VF - Formal Reasoning Artifacts)
    # =====================================================================
    AST_STMT = "AST_STMT"      # 代码语法树（AST）中的语句或块结构
    AST_SYMBOL = "AST_SYMBOL"  # 代码中的标识符、符号、变量或导入项
    SMT_VAR = "SMT_VAR"        # SMT-LIB 逻辑中的变量或一阶逻辑常数
    SMT_ASSERT = "SMT_ASSERT"  # SMT-LIB 逻辑中的断言或约束条件表达式
    SOLVER_OUTCOME = "SOLVER_OUTCOME"  # 外部求解器（Solver）输出结果（SAT / UNSAT / UNKNOWN）

    # =====================================================================
    # L2 受限工具与代理交互层 (VA - Agent Execution Trace)
    # =====================================================================
    STATE = "STATE"            # 环境上下文状态、Prompt 状态、或内部记忆状态
    OBSERVATION = "OBSERVATION" # 来自外部环境（Sandbox/Shell）的观测或反馈文本
    ACTION = "ACTION"          # 代理动作决策或工具调用请求
    TOOL = "TOOL"              # 调用的特定工具、API 或外部接口
    TOOL_RESULT = "TOOL_RESULT" # 工具调用返回的数据、执行日志或标准输出

    # =====================================================================
    # 长期自进化状态图层 (ER-EvoGraph: VH, VE, VM, VP)
    # =====================================================================
    HARNESS_LAYER = "HARNESS_LAYER"     # VH: Harness 层（Prompt模板/重试规则/验证约束）
    VERIFIER_EVIDENCE = "VERIFIER_EVIDENCE" # VE: 验证器证据（Hidden Tests/SMT 证明/反例）
    MEMORY_CELL = "MEMORY_CELL"         # VM: 持久化长/短期记忆、经验槽
    SKILL = "SKILL"                     # VM: 沉淀的代码段、技能或可复用 Workflow
    POLICY_INTERFACE = "POLICY_INTERFACE" # VP: Policy 接口抽象层与工具 Schema 槽位


@unique
class EdgeType(Enum):
    """ISE-Flow 异构有向图中的边关系类型（EdgeType）.
    
    定义了节点之间的逻辑依赖、求值计算、环境演化以及自进化规则准入关系。
    """
    # =====================================================================
    # L1 基础依赖与逻辑边 (Trajectory logical relations)
    # =====================================================================
    SEQUENCE = "SEQUENCE"        # 步骤或事件之间的时序承接关系（默认链式边）
    DERIVE = "DERIVE"            # 步骤到中间命题、claim 的推出/派生关系
    SUPPORT = "SUPPORT"          # 正向支持、逻辑相符或证据增强关系
    CONTRADICT = "CONTRADICT"    # 反向修正、语义矛盾或冲突反驳关系

    # =====================================================================
    # L1 形式化结构边 (Formal/Syntax syntactic relations)
    # =====================================================================
    AST_CHILD = "AST_CHILD"      # Python AST 的父子嵌套关系
    DEPENDENCY = "DEPENDENCY"    # 符号定义与使用链（Use-Def Chain）、异常抛出路径
    CONSTRAINT_DEP = "CONSTRAINT_DEP" # SMT-LIB 断言对底层变量的逻辑依赖
    SOLVER_CHECK = "SOLVER_CHECK" # 约束条件向求解器传递求值的结果指向

    # =====================================================================
    # L2 代理控制流边 (Interactive & Control relations)
    # =====================================================================
    OBSERVE = "OBSERVE"          # 环境观测向状态、推理层的信息输入
    CALL = "CALL"                # Action 对 Tool 的触发/调用关系
    RETURN = "RETURN"            # Tool 执行完毕向 TOOL_RESULT 的返回指向
    TRANSITION = "TRANSITION"    # 动作导致的状态转移
    RECOVER = "RECOVER"          # 工具失败或异常触发的错误处理与回滚关系

    # =====================================================================
    # 长期阶段自进化门控边 (ER-EvoGraph topological constraints)
    # =====================================================================
    ATTRIBUTION = "ATTRIBUTION"  # 失败因果归因边，将执行失败追溯到 Harness 或 Skill 节点
    REGRESSION = "REGRESSION"    # 回归测试依赖边，用于控制持久图变动下的兼容回归
    COMPATIBILITY = "COMPATIBILITY" # 接口协议兼容关系，定义 Skill 与 Policy 槽位之间的映射


@dataclass(frozen=True)
class Node:
    """ISE 轨迹图异构节点实体.

    Attributes:
        node_id: 节点全局唯一标识符（由建图阶段基于哈希或稳定 ID 机制保证）。
        node_type: 节点的异构类别属性（NodeType 枚举）。
        payload: 携带的具体元信息（如文本 span、变量名、AST 解析结果、置信度等）。
        confidence: 该节点的语法置信度或语义自适应概率 p_v，范围在 [0, 1]。
    """
    node_id: str
    node_type: NodeType
    payload: Dict[str, Any] = field(default_factory=dict, hash=False)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """序列化该节点，用于离线存储或 tracker 记录."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "payload": self.payload,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class Edge:
    """ISE 轨迹图异构有向边实体.

    Attributes:
        source_id: 边起点（源节点）的 node_id。
        target_id: 边终点（目标节点）的 node_id。
        edge_type: 边的逻辑或结构化类别属性（EdgeType 枚举）。
        weight: 边权 w_uv，表示逻辑关联强度、语义相似度、或归一化类型权重，默认为 1.0。
        payload: 用于扩展的额外元数据。
    """
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    payload: Dict[str, Any] = field(default_factory=dict, hash=False)

    def to_dict(self) -> Dict[str, Any]:
        """序列化该有向边，用于持久化存储或可视化."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "payload": self.payload,
        }


class HeterogeneousGraphSchema:
    """异构图拓扑 Schema 校验与规则库.
    
    提供对特定 EdgeType 下可允许相连的源节点和目标节点类型的静态约束，
    是 Null-cache 随机图生成和 GraphBuilder 校验防线的核心支撑。
    """
    
    # 静态规则字典：EdgeType -> Set[(SourceNodeType, TargetNodeType)]
    VALID_RELATIONS: Dict[EdgeType, Set[Tuple[NodeType, NodeType]]] = {
        # L1 基础依赖
        EdgeType.SEQUENCE: {
            (NodeType.STEP, NodeType.STEP),
            (NodeType.STEP, NodeType.CLAIM),
            (NodeType.CLAIM, NodeType.STEP),
            (NodeType.STEP, NodeType.ANSWER),
        },
        EdgeType.DERIVE: {
            (NodeType.STEP, NodeType.CLAIM),
            (NodeType.CLAIM, NodeType.CLAIM),
            (NodeType.CLAIM, NodeType.ANSWER),
        },
        EdgeType.SUPPORT: {
            (NodeType.STEP, NodeType.CLAIM),
            (NodeType.VERIFIER_EVIDENCE, NodeType.STEP),
            (NodeType.VERIFIER_EVIDENCE, NodeType.CLAIM),
            (NodeType.VERIFIER_EVIDENCE, NodeType.ANSWER),
        },
        EdgeType.CONTRADICT: {
            (NodeType.STEP, NodeType.STEP),
            (NodeType.CLAIM, NodeType.CLAIM),
            (NodeType.OBSERVATION, NodeType.STEP),
        },

        # L1 形式化 AST / SMT-LIB 依赖
        EdgeType.AST_CHILD: {
            (NodeType.AST_STMT, NodeType.AST_STMT),
            (NodeType.AST_STMT, NodeType.AST_SYMBOL),
        },
        EdgeType.DEPENDENCY: {
            (NodeType.AST_SYMBOL, NodeType.AST_SYMBOL),
            (NodeType.AST_STMT, NodeType.AST_SYMBOL),
        },
        EdgeType.CONSTRAINT_DEP: {
            (NodeType.SMT_ASSERT, NodeType.SMT_VAR),
            (NodeType.SMT_ASSERT, NodeType.SMT_ASSERT),
        },
        EdgeType.SOLVER_CHECK: {
            (NodeType.SMT_ASSERT, NodeType.SOLVER_OUTCOME),
        },

        # L2 交互控制流边
        EdgeType.OBSERVE: {
            (NodeType.OBSERVATION, NodeType.STATE),
            (NodeType.OBSERVATION, NodeType.STEP),
        },
        EdgeType.CALL: {
            (NodeType.ACTION, NodeType.TOOL),
            (NodeType.STEP, NodeType.ACTION),
        },
        EdgeType.RETURN: {
            (NodeType.TOOL, NodeType.TOOL_RESULT),
        },
        EdgeType.TRANSITION: {
            (NodeType.STATE, NodeType.STATE),
            (NodeType.ACTION, NodeType.STATE),
        },
        EdgeType.RECOVER: {
            (NodeType.TOOL_RESULT, NodeType.ACTION),
            (NodeType.TOOL_RESULT, NodeType.STATE),
        },

        # 长期自进化 ER-EvoGraph
        EdgeType.ATTRIBUTION: {
            (NodeType.VERIFIER_EVIDENCE, NodeType.HARNESS_LAYER),
            (NodeType.VERIFIER_EVIDENCE, NodeType.SKILL),
        },
        EdgeType.REGRESSION: {
            (NodeType.HARNESS_LAYER, NodeType.VERIFIER_EVIDENCE),
            (NodeType.SKILL, NodeType.VERIFIER_EVIDENCE),
        },
        EdgeType.COMPATIBILITY: {
            (NodeType.SKILL, NodeType.POLICY_INTERFACE),
            (NodeType.HARNESS_LAYER, NodeType.POLICY_INTERFACE),
        },
    }

    @classmethod
    def validate_connection(cls, src_type: NodeType, dst_type: NodeType, edge_type: EdgeType) -> bool:
        """检验指定的两类节点之间是否存在合法的连接模式.

        Args:
            src_type: 边起点的节点类型.
            dst_type: 边终点的节点类型.
            edge_type: 异构有向边的类型.

        Returns:
            若该连接在 Schema 约束中合法返回 True，否则返回 False.
        """
        allowed_pairs = cls.VALID_RELATIONS.get(edge_type, set())
        return (src_type, dst_type) in allowed_pairs

    @classmethod
    def get_allowed_targets(cls, src_type: NodeType, edge_type: EdgeType) -> List[NodeType]:
        """获取在指定源节点和边类型下允许连接的目标节点类型集合.

        用于引导 Null-cache 进行高可复现性 schema-conditioned 随机重连（rewiring）。
        """
        allowed_pairs = cls.VALID_RELATIONS.get(edge_type, set())
        return [dst for src, dst in allowed_pairs if src == src_type]