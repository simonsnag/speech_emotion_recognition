from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from routers import router
from model import emotion_models
from logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starts...")
    logger.info("Loading model...")
    success = emotion_models.load_backup_model()
    if success:
        logger.info("Backup model loaded successfully")
    else:
        logger.warning("Using default model")

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
