# CoVER-FD Base Detector 集成优先级推荐

集成原则：按**论文实验价值 + 实现成本 + 与 CoVER 适配度**分层集成。目标是证明 CoVER-FD 作为 framework 能泛化到不同 base detector，而不是把所有 GNN 都实现一遍。

---

## 第一优先级：必须有的强 baseline / backbone

### 1. CARE-GNN

专门针对图欺诈检测中的 camouflage fraudster，提出 label-aware similarity、RL neighbor selector、多关系聚合等模块。

与 CoVER 适配度高，可产生自然 evidence：

```text
detector_signal:
  camouflage_neighbor_filter_high
  relation_camouflage_score_high
  selected_neighbor_ratio_low/high

reason_type:
  camouflage_neighbor
  structural_discrepancy
```

集成方式：

```python
BaseModelOutput.extras = {
    "neighbor_selection_score": ...,
    "relation_score": ...,
    "camouflage_score": ...
}
```

对 CoVER 的价值：证明 CoVER 不只适用于 spectral detector BWGNN，也适用于 camouflage-aware fraud detector。

---

### 2. PC-GNN

针对图欺诈检测中的类别不平衡问题，用 label-balanced sampler 和 neighbor selector 进行 "pick and choose" 式训练。

与 scarcity 实验非常匹配，YelpChi / Amazon 本身就是欺诈标签稀缺、不平衡。

可暴露 evidence：

```text
detector_signal:
  minority_neighbor_coverage_high
  imbalance_aware_sampling_signal_high

counter_signal:
  insufficient_minority_neighbor_coverage
```

集成方式：

```python
BaseModelOutput.extras = {
    "sampling_weight": ...,
    "minority_neighbor_coverage": ...,
    "selected_neighbor_count": ...
}
```

对 CoVER 的价值：证明 CoVER 在 class imbalance / label scarcity 下能结合 imbalance-aware detector 的原生信号。

---

### 3. GADBench-style MLP / XGBoost / RF + neighborhood aggregation

GADBench 对 10 个真实图异常检测数据集比较后指出：**简单邻域聚合 + tree ensemble 在监督图异常检测中可以超过不少专门设计的 GNN**。

这对论文非常重要——如果只和 GCN / GAT / SAGE 比，容易被问"为什么不和强 anomaly benchmark baseline 比？"

建议做法：

```text
只作为 detection baseline：
  Raw-XGBoost
  NeighborAgg-XGBoost
  NeighborAgg-RF
```

不建议第一版接入 CoVER reasoner（tree model 没有自然 embeddings / extras），先作为表格 baseline 足够。

---

## 第二优先级：增强泛化说服力的 GNN backbone

### 4. FAGCN

频率自适应 GNN，能融合低频和高频信息。实现成本低，和 BWGNN 一起支撑 spectral / heterophily 叙事。

CoVER evidence：

```text
detector_signal:
  adaptive_high_frequency_response_high
  low_high_frequency_conflict
```

---

### 5. GPR-GNN / BernNet

二选一即可。属于 propagation / spectral filter family，适合处理异配或多频图信号。

- 想实现简单：GPR-GNN
- 想强调频域滤波：BernNet

CoVER evidence：

```text
propagation_disagreement_high
multi_hop_conflict_high
spectral_filter_response_high
```

价值：证明 CoVER 不依赖特定 fraud model，也可适配通用 heterophily/spectral GNN。

---

### 6. H2GCN

适合 heterophily 图场景。图欺诈检测近年也强调 fraud graphs 往往具有 heterophily。

CoVER evidence：

```text
ego_neighbor_conflict_high
two_hop_consistency_high/low
```

时间有限可只作为 baseline，不一定接进 CoVER。

---

## 第三优先级：最新但可选的 fraud-specific 方法

### 7. GHRN

针对 graph anomaly detection 的 heterophily 问题，利用 graph Laplacian 衡量 1-hop label changing / 高频成分以剪枝跨类边。

与 CoVER 结构证据契合：

```text
detector_signal:
  heterophily_indicator_high
  edge_pruning_confidence_high
  high_frequency_label_change_high
```

但复现风险可能较高，建议 BWGNN + CARE-GNN + PC-GNN 稳定后再考虑。

---

### 8. GLC-GNN

针对 fraud detection 中的 heterophily 和 camouflage，提出 global/local confidence 思路。

CoVER evidence：

```text
global_confidence_low
local_confidence_conflict_high
```

不是 YelpChi/Amazon 最经典 baseline，建议"有时间再做"。

---

## 不建议优先集成的模型

| 模型 | 原因 |
|------|------|
| Graph AutoEncoder / DOMINANT / AnomalyDAE | 无监督 / 半监督异常检测方法，主任务是 supervised fake review detection，不适合作为 CoVER base detector |
| GDN | 偏多变量时间序列异常检测，和 YelpChi / Amazon 节点欺诈检测不是同一类任务 |
| GraphLLM / TAPE / LinguGKD / DuoKD 直接作为主任务 backbone | 适合 Cora / PubMed / ogbn-arxiv 的 TAG 语义任务，不建议强行接入主 pipeline。思想已借鉴为 offline teacher cache、support/counter evidence 和 signed distillation |

---

## 推荐的最终集成清单

```text
基础 GNN:
  GCN / GraphSAGE / GAT

强 fraud / anomaly backbone:
  BWGNN / CARE-GNN / PC-GNN

强非 GNN baseline:
  NeighborAgg-XGBoost / NeighborAgg-RF

可选 spectral / heterophily:
  FAGCN 或 GPR-GNN 二选一
```

### 主实验表

| 类别 | 方法 | 接入 CoVER | 作用 |
|------|------|:----------:|------|
| Generic GNN | GCN | 必做 | 基础框架验证 |
| Generic GNN | GraphSAGE | 可选 | 泛化 backbone |
| Generic GNN | GAT | 可选 | attention backbone |
| Fraud GNN | BWGNN | 必做 | 频域异常主 backbone |
| Fraud GNN | CARE-GNN | 推荐 | camouflage fraud backbone |
| Fraud GNN | PC-GNN | 推荐 | imbalance / scarcity backbone |
| Strong ML | NeighborAgg-XGBoost | baseline | GADBench 风格强基线 |
| Strong ML | NeighborAgg-RF | baseline | GADBench 风格强基线 |
| Spectral | FAGCN / GPR-GNN | 可选 | heterophily 泛化 |

### 实现顺序

```text
1. BWGNN
2. NeighborAgg-XGBoost / RF baseline
3. CARE-GNN
4. PC-GNN
5. FAGCN 或 GPR-GNN
6. GHRN / GLC-GNN 可选
```

---

## 对 CoVER 最有价值的三个 backbone

如果只能选三个：

```text
BWGNN + CARE-GNN + PC-GNN
```

分别对应 CoVER reason types：

```text
BWGNN:
  spectral_anomaly

CARE-GNN:
  camouflage_neighbor

PC-GNN:
  weak_or_uncertain_evidence / imbalance-aware structural discrepancy
```

这样 framework 不是泛泛地"支持很多模型"，而是能清楚说明：

```text
不同 base detector 提供不同 detector-native evidence；
CoVER 将它们统一转化为可验证 ERR；
再蒸馏为 LLM-free 可解释预测。
```

这才是最有论文价值的集成方式。
