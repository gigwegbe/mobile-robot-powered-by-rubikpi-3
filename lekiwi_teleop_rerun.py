#!/usr/bin/env python3
"""
LeKiwi Teleop + Rerun Visualisation (Rerun 0.37+)
---------------------------------------------------
Keyboard control of the 3-wheel omni base with live sensor streaming.
Motor control uses the LeRobot FeetechMotorsBus API.

Run on the robot:
    python3 lekiwi_teleop_rerun.py

Install:
    pip install rerun-sdk pyrealsense2 rplidar-roboticia opencv-python numpy
    pip install lerobot   (or have it in your environment already)

Viewing (pick one):
    A) Web browser (nothing to install on laptop):
       Set VIEWER_MODE = "web". Open http://<ROBOT_IP>:9090

    B) Native Rerun viewer on laptop:
       pip install rerun-sdk (on laptop), run: rerun
       Set VIEWER_MODE = "connect" and LAPTOP_IP below.

    C) Local viewer on the robot (needs display):
       Set VIEWER_MODE = "spawn"

Controls:
    W / S     forward / back       (incremental, each press adds speed)
    A / D     strafe left / right
    Q / E     rotate CCW / CW
    Space     stop
    X / Esc   quit
"""

import math
import os
import queue
import select
import signal
import socket
import sys
import termios
import threading
import time
import tty

import cv2
import numpy as np
import pyrealsense2 as rs
import rerun as rr
from rplidar import RPLidar

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

# Global reference so signal handler can stop the LiDAR motor
_lidar = None


# ── Rerun Config ────────────────────────────────────────────────────

VIEWER_MODE = "web"            # "web", "connect", or "spawn"
LAPTOP_IP = "192.168.1.100"    # only for "connect" mode

# ── Motor Config ────────────────────────────────────────────────────

MOTOR_PORT = "/dev/ttyACM0"

WHEELS = {
    "base_left_wheel":  Motor(7, "sts3215", MotorNormMode.RANGE_M100_100),
    "base_back_wheel":  Motor(8, "sts3215", MotorNormMode.RANGE_M100_100),
    "base_right_wheel": Motor(9, "sts3215", MotorNormMode.RANGE_M100_100),
}

WHEEL_RADIUS = 0.05           # m
BASE_RADIUS = 0.125            # m, wheel contact to base centre
MOUNT_ANGLES_DEG = [300.0, 180.0, 60.0]  # left, back, right
MAX_RAW = 3000                 # servo raw speed ceiling
VELOCITY_MODE = 1              # STS3215 Operating_Mode: 1 = velocity

XY_STEP = 0.05                # m/s added per keypress
THETA_STEP = 15.0              # deg/s added per keypress
XY_MAX = 0.30                  # m/s clamp
THETA_MAX = 90.0               # deg/s clamp

# Set False to test sensors without motors connected
MOTORS_ENABLED = True

# ── Sensor Config ───────────────────────────────────────────────────

LIDAR_PORT = "/dev/ttyUSB0"

CAMERA_FPS = 5
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

LOOP_HZ = 30

# ── MiDaS Config ────────────────────────────────────────────────────

MIDAS_ENABLED = True
MIDAS_MODEL_PATH = os.path.expanduser(
    "~/Documents/lekiwi/midas-tflite-w8a8/midas.tflite"
)
MIDAS_INPUT_SIZE = 256
QNN_DELEGATE_PATH = "/usr/lib/libQnnTFLiteDelegate.so"


# ── Kiwi Drive IK ──────────────────────────────────────────────────

def body_to_wheel_raw(vx, vy, omega_degps):
    """Chassis twist (m/s, m/s, deg/s) -> per-wheel raw servo speeds."""
    twist = np.array([vx, vy, math.radians(omega_degps)])
    jac = np.array(
        [
            [math.cos(math.radians(a)), math.sin(math.radians(a)), BASE_RADIUS]
            for a in MOUNT_ANGLES_DEG
        ]
    )
    rim_speeds = jac.dot(twist)
    wheel_degps = np.degrees(rim_speeds / WHEEL_RADIUS)
    raw = wheel_degps * (4096.0 / 360.0)

    peak = np.max(np.abs(raw))
    if peak > MAX_RAW:
        raw = raw * (MAX_RAW / peak)

    return [int(round(r)) for r in raw]


# ── Motor Setup / Teardown ──────────────────────────────────────────

def find_velocity_register(bus):
    """Register name for goal speed varies across LeRobot revisions."""
    probe = next(iter(WHEELS))
    for name in ("Goal_Velocity", "Goal_Speed"):
        try:
            bus.write(name, probe, 0)
            return name
        except Exception:
            continue
    raise RuntimeError(
        "No velocity register found. Check bus.model_ctrl_table."
    )


def close_quietly(bus):
    try:
        bus.disconnect(disable_torque=True)
    except TypeError:
        bus.disconnect()
    except Exception:
        bus.port_handler.closePort()


# ── Keyboard Input ──────────────────────────────────────────────────

def get_key():
    """Non-blocking single keypress (works over SSH)."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1).lower()
    return None


# ── Sensor Streaming Threads ───────────────────────────────────────

def stream_realsense(stop_event):
    """Stream D415 color + depth to Rerun with pinhole intrinsics."""
    print("[D415] Starting...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"[D415] Failed to start: {e}")
        return

    # Log camera intrinsics once (static) for proper 3D depth projection
    try:
        depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_intr = depth_stream.get_intrinsics()
        color_intr = color_stream.get_intrinsics()
        depth_units = profile.get_device().first_depth_sensor().get_depth_scale()

        rr.log("cameras/front", rr.ViewCoordinates.RDF, static=True)

        rr.log(
            "cameras/front/depth",
            rr.Pinhole(
                resolution=[depth_intr.width, depth_intr.height],
                focal_length=[depth_intr.fx, depth_intr.fy],
                principal_point=[depth_intr.ppx, depth_intr.ppy],
            ),
            static=True,
        )

        rr.log(
            "cameras/front/rgb",
            rr.Pinhole(
                resolution=[color_intr.width, color_intr.height],
                focal_length=[color_intr.fx, color_intr.fy],
                principal_point=[color_intr.ppx, color_intr.ppy],
            ),
            static=True,
        )

        print(f"[D415] Depth scale: {depth_units:.6f}")
    except Exception as e:
        print(f"[D415] Could not log intrinsics: {e}")
        depth_units = 0.001  # fallback: 1mm per unit

    interval = 1.0 / CAMERA_FPS
    print("[D415] Streaming")

    try:
        frame_nr = 0
        while not stop_event.is_set():
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                continue  # skip this cycle, try again
            frame_nr += 1

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            try:
                rr.set_time("frame_nr", sequence=frame_nr)
                rr.set_time("scan", sequence=frame_nr)

                if color_frame:
                    img = np.asanyarray(color_frame.get_data())
                    _, jpg = cv2.imencode(
                        ".jpg",
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 50],
                    )
                    rr.log("cameras/front/rgb",
                           rr.EncodedImage(contents=jpg.tobytes(), media_type="image/jpeg"))

                    # Feed latest frame to MiDaS (drop old one if not consumed yet)
                    if MIDAS_ENABLED:
                        try:
                            _midas_frame_queue.put_nowait(img.copy())
                        except queue.Full:
                            try:
                                _midas_frame_queue.get_nowait()
                                _midas_frame_queue.put_nowait(img.copy())
                            except Exception:
                                pass

                if depth_frame:
                    rr.log("cameras/front/depth",
                           rr.DepthImage(np.asanyarray(depth_frame.get_data()),
                                         meter=1.0 / depth_units))
            except Exception:
                pass  # never let logging stall the pipeline

            time.sleep(interval)
    except Exception as e:
        print(f"[D415] Error: {e}")
    finally:
        pipeline.stop()
        print("[D415] Stopped")


def stream_lidar(stop_event):
    """Stream RPLiDAR A2 scans to Rerun as 2D points (matches proven m5.py approach)."""
    print("[LiDAR] Starting...")

    try:
        lidar = RPLidar(LIDAR_PORT, baudrate=115200)
        global _lidar
        _lidar = lidar
        info = lidar.get_info()
        health = lidar.get_health()
        print(f"[LiDAR] info: {info}")
        print(f"[LiDAR] health: {health}")
        lidar.start_motor()
    except Exception as e:
        print(f"[LiDAR] Cannot open {LIDAR_PORT}: {e}")
        return

    # Mark the sensor origin once
    rr.log("lidar/sensor",
           rr.Points2D([[0.0, 0.0]], radii=0.06, colors=[[255, 80, 80]]),
           static=True)

    print("[LiDAR] Streaming")
    scan_count = 0
    max_range = 12000.0  # mm

    try:
        while not stop_event.is_set():
            try:
                for scan in lidar.iter_scans(max_buf_meas=3000, min_len=5):
                    if stop_event.is_set():
                        break

                    ang = np.array([m[1] for m in scan], dtype=np.float32)
                    dist = np.array([m[2] for m in scan], dtype=np.float32)

                    ok = (dist > 0.0) & (dist <= max_range)
                    ang, dist = ang[ok], dist[ok]
                    if ang.size == 0:
                        continue

                    a = np.radians(ang)
                    xy = np.column_stack([dist * np.cos(a),
                                          dist * np.sin(a)]) / 1000.0  # mm -> m

                    # Color by distance: near = red, far = blue
                    t = np.clip(dist / max_range, 0.0, 1.0)
                    colors = np.column_stack([
                        (255 * (1.0 - t)).astype(np.uint8),
                        (60 + 80 * np.sin(np.pi * t)).astype(np.uint8),
                        (255 * t).astype(np.uint8),
                    ])

                    rr.set_time("scan", sequence=scan_count)
                    rr.log("lidar/points",
                           rr.Points2D(xy, radii=0.02, colors=colors))
                    scan_count += 1

            except Exception as e:
                print(f"[LiDAR] {e} -> reconnecting")
                try:
                    lidar.stop()
                    lidar.stop_motor()
                    lidar.disconnect()
                except Exception:
                    pass
                time.sleep(0.5)
                try:
                    lidar = RPLidar(LIDAR_PORT, baudrate=115200)
                    lidar.start_motor()
                except Exception:
                    break

    except Exception as e:
        print(f"[LiDAR] Error: {e}")
    finally:
        try:
            lidar.stop()
            lidar.stop_motor()
            lidar.disconnect()
        except Exception:
            pass
        print("[LiDAR] Stopped")


# Shared queue: D415 thread puts frames here, MiDaS thread reads them
# maxsize=1 means MiDaS always gets the latest frame, never builds a backlog
_midas_frame_queue = queue.Queue(maxsize=1)


def stream_midas(stop_event):
    """Run MiDaS V2 w8a8 on frames from the D415 and log depth to Rerun."""
    print("[MiDaS] Starting...")

    if not os.path.exists(MIDAS_MODEL_PATH):
        print(f"[MiDaS] Model not found: {MIDAS_MODEL_PATH}")
        print("[MiDaS] Download with: qai-hub-models fetch midas -r tflite -p w8a8")
        return

    # Load TFLite interpreter with QNN delegate
    try:
        from ai_edge_litert.interpreter import Interpreter, load_delegate
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter, load_delegate

    delegates = []
    if os.path.exists(QNN_DELEGATE_PATH):
        try:
            delegates.append(load_delegate(QNN_DELEGATE_PATH, options={"backend_type": "htp"}))
            print(f"[MiDaS] QNN delegate loaded (NPU)")
        except Exception as e:
            print(f"[MiDaS] QNN delegate failed: {e}, using CPU")

    interpreter = Interpreter(
        model_path=MIDAS_MODEL_PATH,
        experimental_delegates=delegates if delegates else None,
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_dtype = input_details[0]['dtype']

    print(f"[MiDaS] Input dtype: {input_dtype.__name__}, size: {MIDAS_INPUT_SIZE}x{MIDAS_INPUT_SIZE}")

    # Warmup
    dummy = np.zeros((1, MIDAS_INPUT_SIZE, MIDAS_INPUT_SIZE, 3), dtype=input_dtype)
    interpreter.set_tensor(input_details[0]['index'], dummy)
    for _ in range(3):
        interpreter.invoke()
    print("[MiDaS] Ready")

    midas_count = 0

    while not stop_event.is_set():
        try:
            color_img = _midas_frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        # Preprocess
        img = cv2.resize(color_img, (MIDAS_INPUT_SIZE, MIDAS_INPUT_SIZE))
        if input_dtype == np.uint8:
            img = np.expand_dims(img, axis=0)
        else:
            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            img = np.expand_dims(img, axis=0)

        # Inference
        interpreter.set_tensor(input_details[0]['index'], img)
        t0 = time.perf_counter()
        interpreter.invoke()
        infer_ms = (time.perf_counter() - t0) * 1000

        output = interpreter.get_tensor(output_details[0]['index']).squeeze().astype(np.float32)

        # Normalize to 0-255
        output = output - output.min()
        if output.max() > 0:
            output = output / output.max()
        depth_vis = (output * 255).astype(np.uint8)
        depth_vis = cv2.resize(depth_vis, (CAMERA_WIDTH, CAMERA_HEIGHT))
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        depth_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

        try:
            rr.set_time("scan", sequence=midas_count)
            rr.log("midas/depth", rr.Image(depth_rgb))
            rr.log("perf/midas_ms", rr.Scalars(infer_ms))
        except Exception:
            pass

        midas_count += 1

    print("[MiDaS] Stopped")


# ── Main ───────────────────────────────────────────────────────────

def main():
    print("""
    LeKiwi Teleop + Rerun
    ─────────────────────
    W / S     forward / back
    A / D     strafe left / right
    Q / E     rotate CCW / CW
    Space     stop
    X / Esc   quit
    ─────────────────────
""")

    # ── Init Rerun ──
    rr.init("lekiwi_teleop")

    if VIEWER_MODE == "web":
        # Auto-detect LAN IP so remote browsers can reach the data server
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            host_ip = s.getsockname()[0]
            s.close()
        except Exception:
            host_ip = "127.0.0.1"

        data_uri = f"rerun+http://{host_ip}:9876/proxy"
        rr.serve_grpc(grpc_port=9876, cors_allow_origin=["*"])
        rr.serve_web_viewer(web_port=9090, open_browser=False, connect_to=data_uri)
        full_url = f"http://{host_ip}:9090/?url={data_uri}"
        print(f"[rerun] Open in browser: {full_url}\n")

    elif VIEWER_MODE == "connect":
        rr.connect_grpc(f"rerun+http://{LAPTOP_IP}:9876")
        print(f"[rerun] Connecting to {LAPTOP_IP}:9876\n")
    elif VIEWER_MODE == "spawn":
        rr.spawn()
        print("[rerun] Spawned local viewer\n")

    # ── Init motors ──
    bus = None
    vel_reg = None

    if MOTORS_ENABLED:
        try:
            bus = FeetechMotorsBus(port=MOTOR_PORT, motors=WHEELS)
            bus.connect()
            print(f"[motors] Connected on {MOTOR_PORT}")
            print(f"[motors] IDs: {sorted(m.id for m in WHEELS.values())}")

            bus.disable_torque()
            for name in WHEELS:
                bus.write("Operating_Mode", name, VELOCITY_MODE)
            bus.enable_torque()

            vel_reg = find_velocity_register(bus)
            print(f"[motors] Velocity register: {vel_reg}\n")

        except Exception as e:
            print(f"[motors] Init failed: {e}")
            print("[motors] Running in sensor-only mode.\n")
            bus = None
    else:
        print("[motors] Disabled. Set MOTORS_ENABLED = True when ready.\n")

    # ── Start sensor threads ──
    stop_event = threading.Event()
    threads = [
        threading.Thread(target=stream_realsense, args=(stop_event,), daemon=True),
        threading.Thread(target=stream_lidar, args=(stop_event,), daemon=True),
    ]
    if MIDAS_ENABLED:
        threads.append(
            threading.Thread(target=stream_midas, args=(stop_event,), daemon=True)
        )
    for t in threads:
        t.start()

    time.sleep(2)
    print("Ready. Drive with WASD.\n")

    # ── Teleop loop ──
    vx = vy = omega = 0.0
    period = 1.0 / LOOP_HZ
    old_term = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin.fileno())

        while True:
            key = get_key()

            if key is None:
                pass
            elif key == "w":
                vx = min(vx + XY_STEP, XY_MAX)
            elif key == "s":
                vx = max(vx - XY_STEP, -XY_MAX)
            elif key == "a":
                vy = min(vy + XY_STEP, XY_MAX)
            elif key == "d":
                vy = max(vy - XY_STEP, -XY_MAX)
            elif key == "q":
                omega = min(omega + THETA_STEP, THETA_MAX)
            elif key == "e":
                omega = max(omega - THETA_STEP, -THETA_MAX)
            elif key == " ":
                vx = vy = omega = 0.0
            elif key in ("x", "\x1b"):
                break

            # Compute and send wheel commands
            raw = body_to_wheel_raw(vx, vy, omega)

            if bus and vel_reg:
                for name, value in zip(WHEELS, raw):
                    bus.write(vel_reg, name, value)


            # Terminal feedback
            print(
                f"\rvx={vx:+.2f} vy={vy:+.2f} w={omega:+6.1f}  raw={raw}      ",
                end="", flush=True,
            )
            time.sleep(period)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
        print("\nStopping...")
        stop_event.set()

        # Stop LiDAR motor
        if _lidar:
            try:
                _lidar.stop()
                _lidar.stop_motor()
                _lidar.disconnect()
                print("[LiDAR] Motor stopped.")
            except Exception:
                pass

        if bus and vel_reg:
            try:
                for name in WHEELS:
                    bus.write(vel_reg, name, 0)
                time.sleep(0.1)
            except Exception:
                pass
            close_quietly(bus)

        for t in threads:
            t.join(timeout=3)
        print("Done.")


def _signal_handler(sig, frame):
    """Ensure LiDAR motor stops even on hard kill."""
    if _lidar:
        try:
            _lidar.stop()
            _lidar.stop_motor()
            _lidar.disconnect()
        except Exception:
            pass
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    main()
