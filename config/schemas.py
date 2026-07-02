# -*- coding: utf-8 -*-
"""ISE-Flow 拓扑常数定义.

TSFT 变分场论重构版：移除离散有向图节点，仅保留极简 TOKEN 信息表示。
"""

from enum import Enum, unique

@unique
class NodeType(Enum):
    TOKEN = "TOKEN"

@unique
class EdgeType(Enum):
    ATTENTION = "ATTENTION"