from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.presentation.api.schemas.responses import ApiError


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ValueError)
    async def handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
        request_id = str(uuid4())
        error = ApiError(code="INVALID_INPUT", message=str(exc))

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "request_id": request_id,
                "data": None,
                "error": error.model_dump(),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        request_id = str(uuid4())
        error = ApiError(
            code="REQUEST_VALIDATION_ERROR",
            message="Request validation failed.",
        )

        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "request_id": request_id,
                "data": None,
                "error": error.model_dump(),
            },
        )

    @app.exception_handler(Exception)
    async def handle_generic_error(request: Request, exc: Exception) -> JSONResponse:
        request_id = str(uuid4())
        error = ApiError(
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred.",
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "request_id": request_id,
                "data": None,
                "error": error.model_dump(),
            },
        )