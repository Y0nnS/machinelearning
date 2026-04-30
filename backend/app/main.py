"""
FastAPI Application Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.api.models import router as models_router

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="REST API for ML model inference and management",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(models_router)


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "version": settings.APP_VERSION}


@app.get("/", tags=["system"])
def root():
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "docs": "/docs",
        "version": settings.APP_VERSION,
    }
