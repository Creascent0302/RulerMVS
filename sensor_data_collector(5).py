#!/usr/bin/env python3
"""
CARLA Stereo Reconstruction Data Collector
==========================================
Multi-camera rig for fast stereo 3D reconstruction of a forward-facing zone.

Sensor layout on Tesla Model 3 (x=forward, y=right, z=up):
┌─────────────────────────────────────────────────────────────────┐
│  Stereo group (4 cameras, all converging ~10 m ahead, yaw=0°)  │
│    stereo_up     : (0.0,  0.0, 2.4)  pitch=-10°  roof top     │
│    stereo_down   : (1.0,  0.0, 0.6)  pitch= +5°  front bumper  │
│    stereo_left   : (1.0, -0.9, 1.4)  yaw=+8°     left fender   │
│    stereo_right  : (1.0,  0.9, 1.4)  yaw=-8°     right fender  │
│                                                                 │
│  Telephoto group (2 cameras, narrow FOV 30°, far-field)        │
│    tele_left     : (1.5, -0.3, 1.6)  yaw=0°      front-left    │
│    tele_right    : (1.5,  0.3, 1.6)  yaw=0°      front-right   │
│                                                                 │
│  Depth cameras: one at each RGB mount, with identical FOV    │
│    stereo_up/down/left/right + tele_left/right               │
│                                                                 │
│  Radar (forward-facing, for calibration)                       │
│    radar         : (2.0,  0.0, 1.0)  yaw=0°                    │
└─────────────────────────────────────────────────────────────────┘

Output structure:
  captured_data/
  ├── camera/
  │   ├── stereo_up/       RGB PNGs
  │   ├── stereo_down/
  │   ├── stereo_left/
  │   ├── stereo_right/
  │   ├── tele_left/       telephoto RGB PNGs
  │   └── tele_right/
  ├── depth/<camera>/      per-camera depth .npy (metres, float32)
  ├── preview/             six RGB/depth pairs in one JPG, every 2 seconds
  ├── radar/               radar .npy (x,y,z,velocity)
  └── calibration.json     extrinsics + intrinsics for every sensor
"""

import carla
import numpy as np
import cv2
import os
import time
import json
import math
import logging
import threading
import pygame
from pynput import keyboard as kb_listener

# ======================
# Logging
# ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======================
# Configuration
# ======================
CARLA_HOST = "localhost"
CARLA_PORT = 2000
SAVE_DIR = "captured_data"
CAPTURE_DURATION = 0          # 0 = indefinite (Q to stop)

# ── Spawn position (set to desired map coordinates) ──
# z will be auto-snapped to road surface.
# Set SPAWN_X=None to use default spawn_points[0].
SPAWN_X = 250.0        # meters, map X (0 = origin)
SPAWN_Y = -300.0        # meters, map Y (0 = origin)
SPAWN_YAW = -90.0      # degrees, heading

# Shared resolution for all RGB / depth cameras
IMG_W = 1920
IMG_H = 1080

# Inspection sheets use real elapsed time; raw depth remains full-range metres.
PREVIEW_INTERVAL_SECONDS = 2.0
PREVIEW_DEPTH_MAX_M = 80.0
PREVIEW_TILE_W = 320
PREVIEW_TILE_H = 180
PREVIEW_BUFFER_FRAMES = 16

# ──────────────────────────────────────────────────────────
# Camera definitions
# type: "rgb" | "depth"
# All stereo group cameras share FOV=90 and converge on ~10 m ahead.
# Telephoto cameras use FOV=30 for far-field magnification.
# ──────────────────────────────────────────────────────────
SENSOR_CONFIGS = {
    # ── Stereo group ────────────────────────────────────────
    # Vehicle bounding box extent ≈ (2.4, 1.08, 0.74)
    # All cameras MUST be placed outside this envelope.
    # Rule: pitch must be >= 0 (negative pitch looks down at vehicle roof)
    "stereo_up": {
        "type": "rgb",
        "fov": 90,
        "location": carla.Location(x=2.0, y=0.0, z=2.5),
        "rotation": carla.Rotation(pitch=0, yaw=0, roll=0),
    },
    "stereo_down": {
        "type": "rgb",
        "fov": 90,
        "location": carla.Location(x=3.0, y=0.0, z=1.0),
        "rotation": carla.Rotation(pitch=0, yaw=0, roll=0),
    },
    "stereo_left": {
        "type": "rgb",
        "fov": 90,
        "location": carla.Location(x=1.5, y=-1.5, z=1.5),
        "rotation": carla.Rotation(pitch=0, yaw=12, roll=0),
    },
    "stereo_right": {
        "type": "rgb",
        "fov": 90,
        "location": carla.Location(x=1.5, y=1.5, z=1.5),
        "rotation": carla.Rotation(pitch=0, yaw=-12, roll=0),
    },
    # ── Telephoto group ────────────────────────────────────
    "tele_left": {
        "type": "rgb",
        "fov": 30,
        "location": carla.Location(x=2.8, y=-0.5, z=1.8),
        "rotation": carla.Rotation(pitch=0, yaw=0, roll=0),
    },
    "tele_right": {
        "type": "rgb",
        "fov": 30,
        "location": carla.Location(x=2.8, y=0.5, z=1.8),
        "rotation": carla.Rotation(pitch=0, yaw=0, roll=0),
    },
}

# Keep the original six RGB poses/FOVs; add a depth actor for each RGB actor.
RGB_CAMERA_NAMES = tuple(SENSOR_CONFIGS)
for _camera_name in RGB_CAMERA_NAMES:
    SENSOR_CONFIGS[f"{_camera_name}_depth"] = {
        **SENSOR_CONFIGS[_camera_name], "type": "depth", "camera_name": _camera_name,
    }

# Radar
RADAR_FOV_H = 60.0
RADAR_FOV_V = 30.0
RADAR_RANGE = 100.0
RADAR_LOCATION = carla.Location(x=2.0, y=0.0, z=1.0)
RADAR_ROTATION = carla.Rotation(pitch=0, yaw=0, roll=0)


# ======================
# Directory setup
# ======================

def create_output_dirs(clean=False):
    """Create per-sensor output directories. If clean=True, remove old data first."""
    import shutil
    if clean and os.path.exists(SAVE_DIR):
        # Retry removal (previous sensor callbacks may still be writing)
        for attempt in range(5):
            try:
                shutil.rmtree(SAVE_DIR)
                break
            except OSError:
                time.sleep(0.5)
        logger.info("Cleared old data in %s/", SAVE_DIR)
    for name, cfg in SENSOR_CONFIGS.items():
        sub = "depth" if cfg["type"] == "depth" else "camera"
        os.makedirs(os.path.join(SAVE_DIR, sub, cfg.get("camera_name", name)), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, "radar"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, "preview"), exist_ok=True)
    logger.info("Output directories created under %s/", SAVE_DIR)


# ======================
# Calibration export
# ======================

def _rotation_to_euler(rot):
    """CARLA Rotation → (pitch, yaw, roll) in degrees."""
    return {"pitch": rot.pitch, "yaw": rot.yaw, "roll": rot.roll}


def _compute_intrinsics(fov_deg, width, height):
    """Pinhole intrinsics from horizontal FOV (OpenCV convention)."""
    f = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    return {
        "fx": round(f, 4),
        "fy": round(f, 4),
        "cx": round(cx, 4),
        "cy": round(cy, 4),
        "width": width,
        "height": height,
    }


def save_calibration():
    """Write calibration.json with extrinsics + intrinsics for every sensor."""
    calib = {}
    for name, cfg in SENSOR_CONFIGS.items():
        loc = cfg["location"]
        rot = cfg["rotation"]
        calib[name] = {
            "type": cfg["type"],
            "extrinsic": {
                "x": loc.x, "y": loc.y, "z": loc.z,
                **_rotation_to_euler(rot),
            },
            "intrinsic": _compute_intrinsics(cfg["fov"], IMG_W, IMG_H),
        }
    # Radar
    calib["radar"] = {
        "type": "radar",
        "extrinsic": {
            "x": RADAR_LOCATION.x, "y": RADAR_LOCATION.y, "z": RADAR_LOCATION.z,
            **_rotation_to_euler(RADAR_ROTATION),
        },
        "fov_h": RADAR_FOV_H,
        "fov_v": RADAR_FOV_V,
        "range_m": RADAR_RANGE,
    }

    calib_path = os.path.join(SAVE_DIR, "calibration.json")
    with open(calib_path, "w") as f:
        json.dump(calib, f, indent=2)
    logger.info("Calibration saved → %s", calib_path)
    return calib


# ======================
# Sensor spawning
# ======================

def spawn_vehicle(world):
    """Spawn a Tesla Model 3 at the configured or default position."""
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find("vehicle.tesla.model3")

    if SPAWN_X is not None and SPAWN_Y is not None:
        # User-specified position
        yaw = SPAWN_YAW if SPAWN_YAW is not None else 0.0
        spawn_rotation = carla.Rotation(pitch=0, yaw=yaw, roll=0)
        spawn_transform = carla.Transform(
            carla.Location(x=SPAWN_X, y=SPAWN_Y, z=10.0),  # z=10 temp, will snap
            spawn_rotation
        )
        vehicle = world.spawn_actor(vehicle_bp, spawn_transform)
        spawn_z = 10.0
        logger.info("Spawned at user-specified position (%.1f, %.1f) yaw=%.1f",
                    SPAWN_X, SPAWN_Y, yaw)
    else:
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available")
        vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
        spawn_rotation = spawn_points[0].rotation
        spawn_z = spawn_points[0].location.z
        logger.info("Spawned at default spawn_points[0]")

    time.sleep(0.3)

    # Snap to correct road height (CARLA spawn points often have wrong z)
    loc = vehicle.get_location()
    wp = world.get_map().get_waypoint(loc, project_to_road=True)
    road_z = wp.transform.location.z
    vehicle_half_h = vehicle.bounding_box.extent.z
    correct_z = road_z + vehicle_half_h + 0.1

    vehicle.set_transform(carla.Transform(
        carla.Location(x=loc.x, y=loc.y, z=correct_z),
        spawn_rotation,
    ))
    vehicle.set_simulate_physics(False)   # prevent falling through ground
    vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))

    logger.info("Tesla Model 3 spawned at road z=%.1f (spawn was z=%.1f)",
                correct_z, spawn_z)
    return vehicle


def spawn_sensor(world, vehicle, name, cfg):
    """Spawn one camera (RGB or depth) or return None for unknown type."""
    bp_lib = world.get_blueprint_library()

    if cfg["type"] == "rgb":
        bp = bp_lib.find("sensor.camera.rgb")
    elif cfg["type"] == "depth":
        bp = bp_lib.find("sensor.camera.depth")
    else:
        return None

    bp.set_attribute("image_size_x", str(IMG_W))
    bp.set_attribute("image_size_y", str(IMG_H))
    bp.set_attribute("fov", str(cfg["fov"]))
    bp.set_attribute("sensor_tick", "0.1")   # 10 FPS

    transform = carla.Transform(cfg["location"], cfg["rotation"])
    actor = world.spawn_actor(bp, transform, attach_to=vehicle)
    logger.info(
        "Sensor [%s] (%s, fov=%d°) spawned  id=%s",
        name, cfg["type"], cfg["fov"], actor.id,
    )
    return actor


def spawn_radar(world, vehicle):
    """Spawn forward-facing radar."""
    bp_lib = world.get_blueprint_library()
    radar_bp = bp_lib.find("sensor.other.radar")
    radar_bp.set_attribute("horizontal_fov", str(RADAR_FOV_H))
    radar_bp.set_attribute("vertical_fov", str(RADAR_FOV_V))
    radar_bp.set_attribute("range", str(RADAR_RANGE))
    radar_bp.set_attribute("points_per_second", "10000")
    radar_bp.set_attribute("sensor_tick", "0.1")

    transform = carla.Transform(RADAR_LOCATION, RADAR_ROTATION)
    radar = world.spawn_actor(radar_bp, transform, attach_to=vehicle)
    logger.info("Radar spawned  id=%s", radar.id)
    return radar


# ======================
# Data processing
# ======================

def save_rgb_image(image, cam_name, sim_frame, preview=None):
    """Save an RGB camera image as PNG. Uses CARLA simulation frame number."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]  # BGRA→BGR
    path = os.path.join(SAVE_DIR, "camera", cam_name, f"{cam_name}_{sim_frame:08d}.png")
    cv2.imwrite(path, array)
    if preview is not None:
        preview.add(cam_name, "rgb", sim_frame, array)
    return path


def save_depth_image(image, sensor_name, sim_frame, preview=None):
    """
    Convert CARLA depth image to linear depth (metres) and save as .npy.
    Uses CARLA simulation frame number for synchronization.

    CARLA depth encoding (R, G, B channels):
        linear_depth = (R + G*256 + B*256*256) * 1000 / 16777215   (metres)
    """
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]  # BGRA→BGR
    bgr_f = array.astype(np.float64)
    normalized = (bgr_f[:, :, 2] + bgr_f[:, :, 1] * 256.0 + bgr_f[:, :, 0] * 65536.0) / 16777215.0
    depth_metres = (normalized * 1000.0).astype(np.float32)

    cam_name = SENSOR_CONFIGS[sensor_name]["camera_name"]
    path = os.path.join(SAVE_DIR, "depth", cam_name, f"{cam_name}_{sim_frame:08d}.npy")
    np.save(path, depth_metres)
    if preview is not None:
        preview.add(cam_name, "depth", sim_frame, depth_metres)
    return path


def colorize_depth(depth, max_metres=PREVIEW_DEPTH_MAX_M):
    """Display only: warm=near, cool=far; black marks invalid depth."""
    valid = np.isfinite(depth) & (depth > 0)
    scaled = np.where(valid, np.minimum(depth, max_metres) / max_metres, 0.0)
    levels = np.rint((1.0 - scaled) * 255.0).astype(np.uint8)
    colors = cv2.applyColorMap(levels, cv2.COLORMAP_TURBO)
    colors[~valid] = 0
    return colors


class InspectionPreview:
    """Every two wall seconds, save the latest complete, same-frame sheet.

    Only small thumbnails are retained. A background worker makes sheets so
    the original keyboard/vehicle loop is left intact. Never mix camera frames.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._frames = {}
        self._last_frame = -1
        self._required = {(name, kind) for name in RGB_CAMERA_NAMES for kind in ("rgb", "depth")}

    def add(self, camera_name, kind, sim_frame, array):
        if self._stop.is_set():
            return
        if (camera_name, kind) not in self._required:
            raise ValueError(f"Unknown preview stream {camera_name}.{kind}")
        interpolation = cv2.INTER_NEAREST if kind == "depth" else cv2.INTER_AREA
        thumbnail = cv2.resize(array, (PREVIEW_TILE_W, PREVIEW_TILE_H), interpolation=interpolation)
        with self._lock:
            if sim_frame <= self._last_frame or self._stop.is_set():
                return
            self._frames.setdefault(sim_frame, {})[(camera_name, kind)] = thumbnail
            while len(self._frames) > PREVIEW_BUFFER_FRAMES:
                del self._frames[min(self._frames)]

    def start(self):
        self._thread = threading.Thread(target=self._run, name="inspection-preview", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.wait(PREVIEW_INTERVAL_SECONDS):
            try:
                self.save_latest()
            except Exception:
                logger.exception("Inspection preview failed; collection continues")

    def save_latest(self):
        with self._lock:
            complete = [frame for frame, images in self._frames.items()
                        if images.keys() == self._required and frame > self._last_frame]
            if not complete:
                logger.warning("Preview waiting for six RGB/depth pairs with the same simulation frame")
                return None
            frame = max(complete)
            images = self._frames[frame].copy()
            self._last_frame = frame
            for old in [number for number in self._frames if number <= frame]:
                del self._frames[old]
        mosaic = self._make_mosaic(frame, images)
        path = os.path.join(SAVE_DIR, "preview", f"preview_{frame:08d}.jpg")
        temporary = path + ".tmp.jpg"
        try:
            if not cv2.imwrite(temporary, mosaic, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise OSError(f"Could not write inspection preview {path}")
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        logger.info("Inspection sheet (six same-frame RGB/depth pairs): %s", path)
        return path

    @staticmethod
    def _make_mosaic(frame, images):
        w, h, header, label, footer = PREVIEW_TILE_W, PREVIEW_TILE_H, 62, 28, 68
        mosaic = np.full((header + 2 * (h + label) + footer, 6 * w, 3), 28, np.uint8)
        cv2.putText(mosaic, f"SIM FRAME {frame:08d} | Six RGB + depth pairs",
                    (16, 26), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(mosaic, "Warm = near; cool = far; black = invalid. Raw NPY depth is in metres.",
                    (16, 49), cv2.FONT_HERSHEY_SIMPLEX, .48, (220, 220, 220), 1, cv2.LINE_AA)
        for i, name in enumerate(RGB_CAMERA_NAMES):
            row, col = divmod(i, 3)
            x, y = col * 2 * w, header + row * (h + label)
            cv2.putText(mosaic, name, (x + 8, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, (255, 255, 255), 1, cv2.LINE_AA)
            for kind, image, offset in (
                ("RGB", images[(name, "rgb")], 0),
                ("DEPTH", colorize_depth(images[(name, "depth")]), w),
            ):
                x0, y0 = x + offset, y + label
                mosaic[y0:y0+h, x0:x0+w] = image
                cv2.putText(mosaic, kind, (x0 + 6, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            .5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(mosaic, kind, (x0 + 6, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            .5, (255, 255, 255), 1, cv2.LINE_AA)
        y = header + 2 * (h + label) + 10
        bar = np.tile(np.linspace(255, 0, 500, dtype=np.uint8), (15, 1))
        mosaic[y:y+15, 75:575] = cv2.applyColorMap(bar, cv2.COLORMAP_TURBO)
        cv2.putText(mosaic, "0 m", (16, y+13), cv2.FONT_HERSHEY_SIMPLEX, .45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(mosaic, f"{PREVIEW_DEPTH_MAX_M:g}+ m", (590, y+13), cv2.FONT_HERSHEY_SIMPLEX,
                    .45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(mosaic, "Depth colors use one fixed 0-80 m scale for all cameras and all sheets.",
                    (16, y+43), cv2.FONT_HERSHEY_SIMPLEX, .44, (210, 210, 210), 1, cv2.LINE_AA)
        return mosaic

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


def process_radar_data(radar_measurement, sim_frame):
    """Process and save radar data using CARLA simulation frame number."""
    raw = np.frombuffer(radar_measurement.raw_data, dtype=np.dtype('f4'))
    if raw.size == 0:
        return None

    raw = raw.reshape(-1, 4)
    depths     = raw[:, 0]
    azimuths   = raw[:, 1]
    altitudes  = raw[:, 2]
    velocities = raw[:, 3]

    x = depths * np.cos(altitudes) * np.cos(azimuths)
    y = depths * np.cos(altitudes) * np.sin(azimuths)
    z = depths * np.sin(altitudes)

    points = np.column_stack([x, y, z, velocities])
    path = os.path.join(SAVE_DIR, "radar", f"radar_{sim_frame:08d}.npy")
    np.save(path, points)
    return points


# ======================
# Verification
# ======================

def verify_all(sensor_counters):
    """Validate file counts, image shapes, depth range, radar shape."""
    logger.info("=" * 65)
    logger.info("VERIFICATION — checking captured data integrity")
    logger.info("=" * 65)
    all_ok = True

    for name, expected in sensor_counters.items():
        cfg = SENSOR_CONFIGS.get(name)
        if cfg is None:
            continue  # radar handled separately

        if cfg["type"] == "rgb":
            cam_dir = os.path.join(SAVE_DIR, "camera", name)
            files = sorted(f for f in os.listdir(cam_dir) if f.endswith(".png"))
            count = len(files)
            ok = count == expected
            logger.info("  Camera [%s]: %d frames %s", name, count, "OK" if ok else "MISMATCH")
            if not ok:
                all_ok = False
            # Spot-check first image
            if files:
                img = cv2.imread(os.path.join(cam_dir, files[0]), cv2.IMREAD_UNCHANGED)
                shape_ok = img is not None and img.shape[:2] == (IMG_H, IMG_W)
                logger.info("    first image shape=%s %s",
                            img.shape if img is not None else None,
                            "OK" if shape_ok else "FAIL")
                if not shape_ok:
                    all_ok = False

        elif cfg["type"] == "depth":
            dep_dir = os.path.join(SAVE_DIR, "depth", cfg["camera_name"])
            files = sorted(f for f in os.listdir(dep_dir) if f.endswith(".npy"))
            count = len(files)
            ok = count == expected
            logger.info("  Depth  [%s]: %d frames %s", name, count, "OK" if ok else "MISMATCH")
            if not ok:
                all_ok = False
            if files:
                d = np.load(os.path.join(dep_dir, files[0]))
                valid = d.ndim == 2 and d.dtype == np.float32
                logger.info("    first depth shape=%s dtype=%s range=[%.2f, %.2f] m %s",
                            d.shape, d.dtype, d.min(), d.max(),
                            "OK" if valid else "FAIL")
                if not valid:
                    all_ok = False

    # Radar
    radar_dir = os.path.join(SAVE_DIR, "radar")
    npy_files = sorted(f for f in os.listdir(radar_dir) if f.endswith(".npy"))
    r_expected = sensor_counters.get("radar", 0)
    r_ok = len(npy_files) == r_expected
    logger.info("  Radar: %d frames %s", len(npy_files), "OK" if r_ok else "MISMATCH")
    if not r_ok:
        all_ok = False
    if npy_files:
        sample = np.load(os.path.join(radar_dir, npy_files[0]))
        shape_ok = sample.ndim == 2 and sample.shape[1] == 4
        logger.info("    sample shape=%s (x,y,z,vel) %s", sample.shape, "OK" if shape_ok else "FAIL")
        if not shape_ok:
            all_ok = False

    if all_ok:
        logger.info("✓ VERIFICATION PASSED — all data intact and valid.")
    else:
        logger.warning("✗ VERIFICATION found issues — review warnings above.")
    logger.info("=" * 65)
    return all_ok


def _build_sync_index():
    """
    Build a sync index JSON file mapping simulation frame numbers to sensor files.
    This allows matching data across sensors by the same CARLA sim frame number.

    Output: captured_data/sync_index.json
    Format:
    {
      "<sim_frame>": {
        "stereo_up": "camera/stereo_up/stereo_up_00001234.png",
        "stereo_up_depth": "depth/stereo_up/stereo_up_00001234.npy",
        "radar": "radar/radar_00001234.npy",
        ...
      },
      ...
    }
    """
    import re
    index = {}  # sim_frame_str -> {sensor_name: relative_path}

    # Scan camera directories
    for name in SENSOR_CONFIGS:
        cfg = SENSOR_CONFIGS[name]
        if cfg["type"] == "depth":
            subdir = os.path.join(SAVE_DIR, "depth", cfg["camera_name"])
            prefix = f"{cfg['camera_name']}_"
            ext = ".npy"
        else:
            subdir = os.path.join(SAVE_DIR, "camera", name)
            prefix = f"{name}_"
            ext = ".png"

        if not os.path.isdir(subdir):
            continue
        for fname in os.listdir(subdir):
            if fname.startswith(prefix) and fname.endswith(ext):
                frame_str = fname[len(prefix):-len(ext)]
                index.setdefault(frame_str, {})[name] = os.path.relpath(
                    os.path.join(subdir, fname), SAVE_DIR
                )

    # Scan radar directory
    radar_dir = os.path.join(SAVE_DIR, "radar")
    if os.path.isdir(radar_dir):
        for fname in os.listdir(radar_dir):
            if fname.startswith("radar_") and fname.endswith(".npy"):
                frame_str = fname[6:-4]
                index.setdefault(frame_str, {})["radar"] = os.path.relpath(
                    os.path.join(radar_dir, fname), SAVE_DIR
                )

    # Sort by frame number and write
    sorted_index = dict(sorted(index.items(), key=lambda x: int(x[0])))

    # Count fully-matched frames (all 13 sensors present)
    all_sensors = set(SENSOR_CONFIGS.keys()) | {"radar"}
    matched = sum(1 for v in sorted_index.values() if set(v.keys()) == all_sensors)
    partial = len(sorted_index) - matched

    path = os.path.join(SAVE_DIR, "sync_index.json")
    with open(path, "w") as f:
        json.dump(sorted_index, f, indent=2)

    logger.info("Sync index: %d unique sim frames (%d fully matched, %d partial) → %s",
                len(sorted_index), matched, partial, path)
    return sorted_index


def match_frames_by_name():
    """
    Method 2: Match sensor data by extracting frame numbers from filenames.
    Scans all sensor directories, extracts sim frame numbers, and finds
    frames where all sensors have data (intersection).

    Returns:
        matched_frames: sorted list of sim frame numbers present in ALL sensors
        frame_details: dict {sim_frame: {sensor: filename}}
    """
    import re
    all_sensors = set(SENSOR_CONFIGS.keys()) | {"radar"}
    # frame_str -> {sensor_name: filename}
    frame_map = {}

    # Scan each sensor's directory
    for name in SENSOR_CONFIGS:
        cfg = SENSOR_CONFIGS[name]
        if cfg["type"] == "depth":
            subdir = os.path.join(SAVE_DIR, "depth", cfg["camera_name"])
            pattern = re.compile(rf"^{re.escape(cfg['camera_name'])}_(\d+)\.npy$")
        else:
            subdir = os.path.join(SAVE_DIR, "camera", name)
            pattern = re.compile(rf"^{name}_(\d+)\.png$")

        if not os.path.isdir(subdir):
            continue
        for fname in os.listdir(subdir):
            m = pattern.match(fname)
            if m:
                frame_str = m.group(1)
                frame_map.setdefault(frame_str, {})[name] = fname

    # Scan radar
    radar_dir = os.path.join(SAVE_DIR, "radar")
    if os.path.isdir(radar_dir):
        for fname in os.listdir(radar_dir):
            m = re.match(r"^radar_(\d+)\.npy$", fname)
            if m:
                frame_str = m.group(1)
                frame_map.setdefault(frame_str, {})["radar"] = fname

    # Find fully matched frames (all 13 sensors present)
    matched_frames = []
    partial_frames = []
    for frame_str, sensors in frame_map.items():
        if set(sensors.keys()) == all_sensors:
            matched_frames.append(int(frame_str))
        else:
            partial_frames.append(int(frame_str))

    matched_frames.sort()
    partial_frames.sort()

    # Build detail dict for matched frames
    frame_details = {}
    for frame_num in matched_frames:
        frame_str = f"{frame_num:08d}" if frame_num < 100000000 else str(frame_num)
        # Try both zero-padded and raw
        key = frame_str if frame_str in frame_map else str(frame_num)
        if key in frame_map:
            frame_details[frame_num] = frame_map[key]

    # Log summary
    logger.info("=" * 65)
    logger.info("FILENAME MATCHING — matching by frame number in filenames")
    logger.info("=" * 65)
    logger.info("  Total unique frame numbers found: %d", len(frame_map))
    logger.info("  Fully matched (all %d sensors): %d frames",
                len(all_sensors), len(matched_frames))
    logger.info("  Partial (missing some sensors): %d frames", len(partial_frames))

    if matched_frames:
        logger.info("  Matched frame range: %d — %d",
                    matched_frames[0], matched_frames[-1])

    if partial_frames:
        # Show which sensors are missing in partial frames
        missing_summary = {}
        for frame_num in partial_frames:
            key = str(frame_num)
            # Try zero-padded too
            for k in [str(frame_num), f"{frame_num:08d}"]:
                if k in frame_map:
                    key = k
                    break
            present = set(frame_map.get(key, {}).keys())
            missing = all_sensors - present
            missing_key = ",".join(sorted(missing))
            missing_summary[missing_key] = missing_summary.get(missing_key, 0) + 1
        logger.info("  Partial frame breakdown (missing sensors → count):")
        for sensors_missing, count in sorted(missing_summary.items()):
            logger.info("    missing [%s]: %d frames", sensors_missing, count)

    # Save matched frame list
    matched_path = os.path.join(SAVE_DIR, "matched_frames.json")
    with open(matched_path, "w") as f:
        json.dump({
            "matched_count": len(matched_frames),
            "partial_count": len(partial_frames),
            "total_sensors": len(all_sensors),
            "matched_frames": matched_frames,
            "frame_details": {str(k): v for k, v in frame_details.items()},
        }, f, indent=2)
    logger.info("Matched frames saved → %s", matched_path)
    logger.info("=" * 65)

    return matched_frames, frame_details


# ======================
# Main
# ======================

def run():
    create_output_dirs(clean=True)

    # Connect
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)
    world = client.get_world()
    logger.info("Connected to CARLA, map: %s", world.get_map().name)

    # Clean up any leftover vehicles/sensors from previous runs
    for actor in world.get_actors().filter('vehicle.*'):
        try:
            actor.destroy()
        except Exception:
            pass
    for actor in world.get_actors().filter('sensor.*'):
        try:
            actor.destroy()
        except Exception:
            pass
    time.sleep(0.5)
    logger.info("Cleaned up leftover actors from previous runs.")

    # Save calibration before spawning anything
    calib = save_calibration()

    # Spawn vehicle
    vehicle = spawn_vehicle(world)

    # Spawn all sensors
    sensors = {}   # name → actor
    for name, cfg in SENSOR_CONFIGS.items():
        sensors[name] = spawn_sensor(world, vehicle, name, cfg)

    radar = spawn_radar(world, vehicle)

    # Frame counters
    counters = {name: 0 for name in sensors}
    counters["radar"] = 0
    preview = InspectionPreview()

    # ── Callbacks (use CARLA simulation frame number for sync) ────
    def make_rgb_callback(cam_name):
        def on_image(image):
            sim_frame = image.frame
            counters[cam_name] += 1
            save_rgb_image(image, cam_name, sim_frame, preview=preview)
            if counters[cam_name] % 10 == 0:
                logger.info("[%s] frame %d  (sim=%d, %dx%d)",
                            cam_name, counters[cam_name], sim_frame, image.width, image.height)
        return on_image

    def make_depth_callback(cam_name):
        def on_image(image):
            sim_frame = image.frame
            counters[cam_name] += 1
            save_depth_image(image, cam_name, sim_frame, preview=preview)
            if counters[cam_name] % 10 == 0:
                logger.info("[%s] frame %d  (sim=%d) depth saved", cam_name, counters[cam_name], sim_frame)
        return on_image

    def on_radar(data):
        sim_frame = data.frame
        counters["radar"] += 1
        result = process_radar_data(data, sim_frame)
        if counters["radar"] % 10 == 0 and result is not None:
            logger.info("[radar] frame %d  (sim=%d, %d pts)", counters["radar"], sim_frame, len(result))

    # Register listeners
    for name, actor in sensors.items():
        cfg = SENSOR_CONFIGS[name]
        if cfg["type"] == "rgb":
            actor.listen(make_rgb_callback(name))
        elif cfg["type"] == "depth":
            actor.listen(make_depth_callback(name))
    radar.listen(on_radar)
    preview.start()

    total_sensors = len(sensors) + 1  # +1 for radar
    logger.info("All %d sensor listeners registered. Capturing...", total_sensors)

    # ── Keyboard control (main thread) ──────────────────────
    vehicle.set_autopilot(False)
    logger.info("Keyboard: ↑↓=drive  ←→=steer  SPACE=stop  V=switch view  Q=quit")
    logger.info("(keyboard works globally — no need to click any window)")

    # Spectator view modes: 0=off  1=third-person  2=first-person  3=top-down
    _view_mode = 1
    _view_names = ["Free", "Third-person", "First-person", "Top-down"]
    spectator = world.get_spectator()

    # Vehicle state
    _speed = 0.0        # m/s
    _yaw_deg = vehicle.get_transform().rotation.yaw
    _max_speed = 30.0   # m/s (~108 km/h)
    _accel = 6.0        # m/s²
    _brake = 10.0       # m/s²
    _steer_rate = 40.0  # deg/s
    _friction = 2.0     # m/s² natural decel

    # ── Global keyboard state (pynput) ──
    _keys_pressed = set()
    _quit_requested = False
    _view_switch_requested = False

    _KEY_MAP = {
        kb_listener.Key.up: "UP",
        kb_listener.Key.down: "DOWN",
        kb_listener.Key.left: "LEFT",
        kb_listener.Key.right: "RIGHT",
        kb_listener.Key.space: "SPACE",
    }

    def on_press(key):
        nonlocal _quit_requested, _view_switch_requested
        logger.info("[KB] pressed: %s (type=%s)", key, type(key).__name__)
        name = _KEY_MAP.get(key)
        if name:
            _keys_pressed.add(name)
            logger.info("[KB] mapped to: %s", name)
        elif key == kb_listener.KeyCode.from_char('q'):
            _quit_requested = True
        elif key == kb_listener.KeyCode.from_char('v'):
            _view_switch_requested = True

    def on_release(key):
        name = _KEY_MAP.get(key)
        if name:
            _keys_pressed.discard(name)
            logger.info("[KB] released: %s", name)

    kb_thread = kb_listener.Listener(on_press=on_press, on_release=on_release)
    kb_thread.start()
    logger.info("Global keyboard listener started.")

    # Background: snap vehicle z to road surface
    def keep_on_road():
        nav_map = world.get_map()
        while True:
            try:
                cur = vehicle.get_location()
                wp = nav_map.get_waypoint(cur, project_to_road=True)
                vh = vehicle.bounding_box.extent.z
                road_z = wp.transform.location.z + vh + 0.1
                if abs(cur.z - road_z) > 0.5:
                    t = vehicle.get_transform()
                    vehicle.set_transform(carla.Transform(
                        carla.Location(x=t.location.x, y=t.location.y, z=road_z),
                        t.rotation
                    ))
                time.sleep(0.5)
            except Exception:
                break

    road_thread = threading.Thread(target=keep_on_road, daemon=True)
    road_thread.start()

    # Init pygame on main thread (HUD only, no keyboard)
    pygame.init()
    screen = pygame.display.set_mode((400, 120))
    pygame.display.set_caption("CARLA HUD")
    font = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()
    start_time = time.time()
    last_status = 0
    _debug_key_timer = 0

    try:
        while True:
            dt = clock.tick(30) / 1000.0  # 30 Hz control loop

            # ── Check quit ──
            if _quit_requested:
                raise KeyboardInterrupt

            # ── Check view switch ──
            if _view_switch_requested:
                _view_switch_requested = False
                _view_mode = (_view_mode + 1) % len(_view_names)
                logger.info("View → %s", _view_names[_view_mode])

            # ── Debug key state ──
            _debug_key_timer += dt
            if _debug_key_timer > 2.0:
                _debug_key_timer = 0
                logger.info("[DEBUG] speed=%.2f yaw=%.1f keys=%s", _speed, _yaw_deg, _keys_pressed)

            # ── Acceleration / braking ──
            if "UP" in _keys_pressed:
                _speed = min(_speed + _accel * dt, _max_speed)
            elif "DOWN" in _keys_pressed:
                _speed = max(_speed - _brake * dt, -_max_speed * 0.3)
            else:
                if _speed > 0:
                    _speed = max(0, _speed - _friction * dt)
                elif _speed < 0:
                    _speed = min(0, _speed + _friction * dt)
            if "SPACE" in _keys_pressed:
                _speed = 0.0

            # ── Steering ──
            if "LEFT" in _keys_pressed:
                _yaw_deg += _steer_rate * dt
            if "RIGHT" in _keys_pressed:
                _yaw_deg -= _steer_rate * dt

            # ── Move vehicle ──
            if abs(_speed) > 0.01:
                yaw_rad = math.radians(_yaw_deg)
                cur = vehicle.get_location()
                new_x = cur.x + _speed * math.cos(yaw_rad) * dt
                new_y = cur.y + _speed * math.sin(yaw_rad) * dt
                vehicle.set_transform(carla.Transform(
                    carla.Location(x=new_x, y=new_y, z=cur.z),
                    carla.Rotation(pitch=0, yaw=_yaw_deg, roll=0)
                ))

            # ── Spectator view ──
            veh_t = vehicle.get_transform()
            veh_loc = veh_t.location
            if _view_mode == 1:  # third-person
                sp_loc = carla.Location(
                    x=veh_loc.x - 8 * math.cos(math.radians(_yaw_deg)),
                    y=veh_loc.y - 8 * math.sin(math.radians(_yaw_deg)),
                    z=veh_loc.z + 4
                )
                spectator.set_transform(carla.Transform(
                    sp_loc,
                    carla.Rotation(pitch=-15, yaw=_yaw_deg, roll=0)
                ))
            elif _view_mode == 2:  # first-person
                sp_loc = carla.Location(
                    x=veh_loc.x + 1.5 * math.cos(math.radians(_yaw_deg)),
                    y=veh_loc.y + 1.5 * math.sin(math.radians(_yaw_deg)),
                    z=veh_loc.z + 1.6
                )
                spectator.set_transform(carla.Transform(
                    sp_loc,
                    carla.Rotation(pitch=0, yaw=_yaw_deg, roll=0)
                ))
            elif _view_mode == 3:  # top-down
                spectator.set_transform(carla.Transform(
                    carla.Location(x=veh_loc.x, y=veh_loc.y, z=veh_loc.z + 50),
                    carla.Rotation(pitch=-90, yaw=0, roll=0)
                ))

            # ── HUD ──
            screen.fill((30, 30, 30))
            spd_txt = font.render(f"Speed: {_speed:.1f} m/s  ({_speed*3.6:.0f} km/h)", True, (0,255,0))
            yaw_txt = font.render(f"Yaw: {_yaw_deg:.1f} deg", True, (0,255,0))
            view_txt = font.render(f"View: {_view_names[_view_mode]}  (V to switch)", True, (0,200,255))
            hint = font.render("Arrows:drive  Space:stop  Q:quit", True, (180,180,180))
            screen.blit(spd_txt, (10, 10))
            screen.blit(yaw_txt, (10, 30))
            screen.blit(view_txt, (10, 50))
            screen.blit(hint, (10, 75))
            pygame.display.flip()

            # ── Status log ──
            elapsed = time.time() - start_time
            if int(elapsed) % 5 == 0 and int(elapsed) > 0 and int(elapsed) != last_status:
                last_status = int(elapsed)
                parts = [f"{n}: {c}" for n, c in counters.items() if c > 0]
                logger.info("T=%ds  %s", int(elapsed), " | ".join(parts))

            if CAPTURE_DURATION > 0 and elapsed >= CAPTURE_DURATION:
                logger.info("Capture duration (%ds) reached.", CAPTURE_DURATION)
                break

    except KeyboardInterrupt:
        logger.info("User stopped collection.")

    finally:
        preview.close()
        try:
            kb_thread.stop()
            pygame.quit()
            # ── Cleanup ──────────────────────────────────────────
            logger.info("Cleaning up all sensors and vehicle...")

            # 1) Stop all sensor listeners first
            for actor in sensors.values():
                try:
                    actor.stop()
                except Exception:
                    pass
            try:
                radar.stop()
            except Exception:
                pass

            # 2) Wait for in-flight callbacks to drain
            time.sleep(0.5)

            # 3) Destroy sensor actors (they depend on vehicle, so destroy first)
            for actor in sensors.values():
                try:
                    if actor.is_alive:
                        actor.destroy()
                except Exception:
                    pass
            try:
                if radar.is_alive:
                    radar.destroy()
            except Exception:
                pass

            # 4) Brief pause so sensor destruction propagates
            time.sleep(0.3)

            # 5) Finally destroy the vehicle
            try:
                if vehicle.is_alive:
                    vehicle.destroy()
            except Exception:
                pass

            # ── Summary ──────────────────────────────────────────
            total_frames = sum(counters.values())
            breakdown = ", ".join(f"{n}={c}" for n, c in counters.items())
            logger.info(
                "Capture complete — %d total frames (%s). Data in %s/",
                total_frames, breakdown, SAVE_DIR,
            )

            # ── Verification ─────────────────────────────────────
            verify_all(counters)

        except Exception as e:
            logger.warning("Cleanup interrupted: %s", e)

        # ── Data matching (always run, even if cleanup was interrupted) ──
        try:
            logger.info("Building sync index...")
            _build_sync_index()
        except Exception as e:
            logger.warning("Sync index build failed: %s", e)

        try:
            logger.info("Matching frames by filename...")
            match_frames_by_name()
        except Exception as e:
            logger.warning("Filename matching failed: %s", e)


if __name__ == "__main__":
    run()

