from typing import List, Optional
from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    history: Optional[List[str]] = Field(None, description="Recent user history item IDs")
    num_recommendations: int = Field(20, ge=1, le=100)
    exclude_ids: Optional[List[str]] = Field(None, description="Item IDs to exclude")


class RecommendationItem(BaseModel):
    item_id: str
    score: float
    rank: int


class RecommendResponse(BaseModel):
    user_id: str
    recommendations: List[RecommendationItem]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    index_size: int
    cache_connected: bool


class BatchRecommendRequest(BaseModel):
    requests: List[RecommendRequest]


class BatchRecommendResponse(BaseModel):
    responses: List[RecommendResponse]
    total_latency_ms: float
