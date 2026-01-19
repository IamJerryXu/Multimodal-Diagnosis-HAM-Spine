from .fusion_blocks import (
    FusionModule,
    ConcatFusionModule,
    MultiScaleFusionModule,
    WeightedConcatFusionModule,
    HadamardFusionModule,
    BilinearFusionModule,
    SSMFusionModule,
    VMambaFusionModule,
)
from .heads import ResidualClassifier, AttentionPoolingClassifier, build_kan_head
from .gating import DualExpertGate
from .tabular import TabularEncoder

__all__ = [
    "FusionModule",
    "ConcatFusionModule",
    "MultiScaleFusionModule",
    "WeightedConcatFusionModule",
    "HadamardFusionModule",
    "BilinearFusionModule",
    "SSMFusionModule",
    "VMambaFusionModule",
    "ResidualClassifier",
    "AttentionPoolingClassifier",
    "build_kan_head",
    "DualExpertGate",
    "TabularEncoder",
]
