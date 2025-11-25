"""Model implementations for the recommendation system."""

from .two_tower import TwoTowerModel, UserTower, ItemTower, create_two_tower_model

# Optional ranking model imports (requires xgboost/lightgbm)
try:
    from .ranking_models import RankingModel, XGBoostRanker, DeepFMRanker, create_ranking_model
    _RANKING_AVAILABLE = True
except ImportError:
    _RANKING_AVAILABLE = False
    RankingModel = None
    XGBoostRanker = None
    DeepFMRanker = None
    create_ranking_model = None

__all__ = [
    "TwoTowerModel",
    "UserTower",
    "ItemTower",
    "create_two_tower_model",
]

if _RANKING_AVAILABLE:
    __all__.extend(["RankingModel", "XGBoostRanker", "DeepFMRanker", "create_ranking_model"])