from fastapi import APIRouter, UploadFile, File
from app.services.PowerIndex.ankle_rf_predictor_powerindex_services import predict_ankle_power
import os

router = APIRouter()

@router.post("/power/ankle_rf_power",tags=["Power Index"])
async def get_rf_power(
    foot_file: UploadFile = File(...),
    shank_file: UploadFile = File(...)
):
    upload_dir = "uploaded_files"
    os.makedirs(upload_dir, exist_ok=True)

    foot_path = os.path.join(upload_dir, foot_file.filename)
    shank_path = os.path.join(upload_dir, shank_file.filename)

    with open(foot_path, "wb") as f:
        f.write(await foot_file.read())
    with open(shank_path, "wb") as s:
        s.write(await shank_file.read())

    result = predict_ankle_power(foot_path, shank_path)

    os.remove(foot_path)
    os.remove(shank_path)

    return result
