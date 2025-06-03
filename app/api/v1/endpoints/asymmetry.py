from fastapi import APIRouter, UploadFile, File
from app.services.asymmetry_services import asymmetry_analysis, asymmetry_ankleAngles,asymmetry_kneeAngles
import os
import numpy as np

router = APIRouter()

@router.post("/asymmetry/gait_metrics")
async def get_gait_metrics(file: UploadFile = File(...)):
    upload_dir = "uploaded_files"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, file.filename)
    
    # Save file to disk
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Now call the function with file path just like your original usage
    df_out, metrics = asymmetry_analysis.compute_power_metrics_with_zupt(
        filepaths=[file_path],
        foot_mass=1.0,
        foot_inertia_diag=(0.0010, 0.0053, 0.0060),
        use_quaternion=True,
        remove_gravity=True,
        apply_filter=True,
        fs=100.0,
        cutoff=10.0,
        filter_order=4,
        stationary_time_s=2.0,
        body_mass=65.0,
        plot_results=False
    )

    return {
        "metrics": {k: v.tolist() if hasattr(v, "tolist") else v for k, v in metrics.items()}
    }
@router.post("/asymmetry/ankleAngles")
async def get_joint_angles(
    shank_file: UploadFile = File(...),
    foot_file: UploadFile = File(...)
):
    upload_dir = "uploaded_files"
    os.makedirs(upload_dir, exist_ok=True)

    shank_path = os.path.join(upload_dir, shank_file.filename)
    foot_path = os.path.join(upload_dir, foot_file.filename)

    with open(shank_path, "wb") as s:
        s.write(await shank_file.read())
    with open(foot_path, "wb") as f:
        f.write(await foot_file.read())

    # Run analysis
    time_s, (_, angles_rot, angles_flex), (_, rot_filt, flex_filt), *_ = asymmetry_ankleAngles.analyze_two_imus(
        shank_path,
        foot_path,
        standing_time_s=5.0,
        ignore_initial_s=5.0,
        lowpass_cutoff=0.15,
        margin_before=15,
        margin_after=30,
        required_increase_ratio=1.2,
        cycle_length=101,
        cycle_to_remove=None
    )

    # Cleanup files
    os.remove(shank_path)
    os.remove(foot_path)

    # Prepare filtered metrics only
    return {
        "filtered_metrics": {
            "eversion_angle_deg": {
                "max": float(np.max(rot_filt)),
                "min": float(np.min(rot_filt)),
                "avg": float(np.mean(rot_filt))
            },
            "rotation_angle_deg": {
                "max": float(np.max(flex_filt)),
                "min": float(np.min(flex_filt)),
                "avg": float(np.mean(flex_filt))
            }
        }
    }

@router.post("/asymmetry/kneeAngles")
async def get_knee_angles(
    thigh_file: UploadFile = File(...),
    shank_file: UploadFile = File(...)
):
    upload_dir = "uploaded_files"
    os.makedirs(upload_dir, exist_ok=True)

    thigh_path = os.path.join(upload_dir, thigh_file.filename)
    shank_path = os.path.join(upload_dir, shank_file.filename)

    with open(thigh_path, "wb") as tf:
        tf.write(await thigh_file.read())
    with open(shank_path, "wb") as sf:
        sf.write(await shank_file.read())

    # Run analysis
    time_s, (_, _, _), (flex_filt, var_filt, int_ext_filt), *_ = asymmetry_kneeAngles.analyze_two_imus(
        filename_thigh=thigh_path,
        filename_shank=shank_path,
        standing_time_s=5.0,
        ignore_initial_s=5.0,
        lowpass_cutoff=0.15,
        margin_before=15,
        margin_after=30,
        required_increase_ratio=1.2,
        cycle_length=101,
        cycle_to_remove=None
    )

    os.remove(thigh_path)
    os.remove(shank_path)

    return {
        "knee_angles_filtered": {
            "flexion_extension": {
                "max": float(np.max(flex_filt)),
                "min": float(np.min(flex_filt)),
                "avg": float(np.mean(flex_filt))
            },
            "varus_valgus": {
                "max": float(np.max(var_filt)),
                "min": float(np.min(var_filt)),
                "avg": float(np.mean(var_filt))
            },
            "internal_external_rotation": {
                "max": float(np.max(int_ext_filt)),
                "min": float(np.min(int_ext_filt)),
                "avg": float(np.mean(int_ext_filt))
            }
        }
    }
