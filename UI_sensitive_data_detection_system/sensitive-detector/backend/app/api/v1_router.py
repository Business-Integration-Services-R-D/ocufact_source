from fastapi import APIRouter
from .endpoints import router as endpoints_router

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(endpoints_router)

