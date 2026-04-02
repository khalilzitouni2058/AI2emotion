from typing import Any, Optional

from pydantic import BaseModel, Field


class ApiError(BaseModel):
    """Standard error payload returned by the API."""

    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")


class ApiResponse(BaseModel):
    """Standard wrapper for all API responses."""

    success: bool = Field(..., description="Whether the request succeeded")
    request_id: str = Field(..., description="Unique request identifier")
    data: Optional[Any] = Field(default=None, description="Successful response payload")
    error: Optional[ApiError] = Field(default=None, description="Error payload if request failed")