from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from routers import router
from logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starts...")
    yield
    logger.info("Application shuts down...")


app = FastAPI(
    title="Emotion Recognition API",
    description="API emotion recognition",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "Emotion Recognition API",
        "docs_url": "/docs",
        "status": "online",
    }
