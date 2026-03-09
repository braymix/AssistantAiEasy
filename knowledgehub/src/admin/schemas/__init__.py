"""Admin API schemas."""

from src.admin.schemas.analytics import (
    ConversationTrend,
    ContextUsage,
    DailyCount,
    KnowledgeGrowth,
    OverviewStats,
    RulePerformance,
)
from src.admin.schemas.common import ApiResponse, ErrorDetail, PaginationMeta
from src.admin.schemas.contexts import (
    ContextCreate,
    ContextKnowledgeItem,
    ContextOut,
    ContextStats,
    ContextUpdate,
)
from src.admin.schemas.knowledge import (
    BulkDocumentInput,
    BulkImportRequest,
    BulkImportResponse,
    ExportItem,
    ExportResponse,
    KnowledgeItemOut,
    VerifyAction,
)
from src.admin.schemas.rules import (
    ReloadResponse,
    RuleCreate,
    RuleOut,
    RuleTestRequest,
    RuleTestResult,
    RuleUpdate,
)

__all__ = [
    "ApiResponse",
    "BulkDocumentInput",
    "BulkImportRequest",
    "BulkImportResponse",
    "ContextCreate",
    "ContextKnowledgeItem",
    "ContextOut",
    "ContextStats",
    "ContextUpdate",
    "ContextUsage",
    "ConversationTrend",
    "DailyCount",
    "ErrorDetail",
    "ExportItem",
    "ExportResponse",
    "KnowledgeGrowth",
    "KnowledgeItemOut",
    "OverviewStats",
    "PaginationMeta",
    "ReloadResponse",
    "RuleCreate",
    "RuleOut",
    "RulePerformance",
    "RuleTestRequest",
    "RuleTestResult",
    "RuleUpdate",
    "VerifyAction",
]
