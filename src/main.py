from fastapi import FastAPI
from app.api.v1.endpoints.asymmetry import router as gait_router
from app.api.v1.endpoints.powerIndex_ankle_rf import router as powerIndex_ankle_rf

app = FastAPI(
    title="upLYFT Macro View Screens API",
    description="Upload IMU CSV data to compute gait metrics and power analytics",
    version="1.0.0"
)

app.include_router(gait_router, prefix="/api/v1", tags=["Asymmetry Index"])
app.include_router(powerIndex_ankle_rf, prefix="/api/v1", tags=["Power Index"])
