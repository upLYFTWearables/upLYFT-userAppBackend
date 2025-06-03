from fastapi import APIRouter
from .asymmetry import router as gait_router  # <-- add this line
from .powerIndex_ankle_rf import router as powerIndexAnkle_rf

router = APIRouter()
router.include_router(gait_router, prefix="/asymmetry", tags=["Asymmetry Index"])  # <-- add this line
router.include_router(powerIndexAnkle_rf, prefix="/power", tags=["Power Index"])
