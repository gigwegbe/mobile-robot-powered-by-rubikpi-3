#!/usr/bin/env python3
"""
LeKiwi Web Teleop
-----------------
Touch/mouse joystick control from a browser (laptop or iPad).
Runs alongside the Rerun visualisation script.

Install:
    pip install fastapi uvicorn websockets

Run:
    python3 lekiwi_web_teleop.py

Then open http://<ROBOT_IP>:8000 on your laptop or iPad.
"""

import asyncio
import json
import math

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

# ── Motor Config (same as teleop script) ───────────────────────────

MOTOR_PORT = "/dev/ttyACM0"

WHEELS = {
    "base_left_wheel":  Motor(7, "sts3215", MotorNormMode.RANGE_M100_100),
    "base_back_wheel":  Motor(8, "sts3215", MotorNormMode.RANGE_M100_100),
    "base_right_wheel": Motor(9, "sts3215", MotorNormMode.RANGE_M100_100),
}

WHEEL_RADIUS = 0.05
BASE_RADIUS = 0.125
MOUNT_ANGLES_DEG = [300.0, 180.0, 60.0]
MAX_RAW = 3000
VELOCITY_MODE = 1

XY_MAX = 0.30       # m/s
THETA_MAX = 90.0     # deg/s


# ── Kiwi Drive IK ─────────────────────────────────────────────────

def body_to_wheel_raw(vx, vy, omega_degps):
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


# ── Motor Setup ────────────────────────────────────────────────────

def find_velocity_register(bus):
    probe = next(iter(WHEELS))
    for name in ("Goal_Velocity", "Goal_Speed"):
        try:
            bus.write(name, probe, 0)
            return name
        except Exception:
            continue
    raise RuntimeError("No velocity register found.")


def close_quietly(bus):
    try:
        bus.disconnect(disable_torque=True)
    except TypeError:
        bus.disconnect()
    except Exception:
        bus.port_handler.closePort()


# ── Web UI HTML ────────────────────────────────────────────────────

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
<title>LeKiwi Teleop</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/nipplejs/0.10.2/nipplejs.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: #1a1a2e;
    color: #e0e0e0;
    font-family: 'SF Mono', 'Fira Code', monospace;
    height: 100vh;
    overflow: hidden;
    touch-action: none;
    display: flex;
    flex-direction: column;
  }

  .header {
    text-align: center;
    padding: 12px;
    background: #16213e;
    border-bottom: 1px solid #0f3460;
  }

  .header h1 {
    font-size: 16px;
    font-weight: 500;
    letter-spacing: 2px;
    color: #e94560;
  }

  .status {
    display: flex;
    justify-content: center;
    gap: 24px;
    padding: 10px;
    font-size: 13px;
    background: #16213e;
    border-bottom: 1px solid #0f3460;
  }

  .status span { opacity: 0.6; }
  .status strong { color: #e94560; font-weight: 600; }

  .controls {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px;
    gap: 20px;
  }

  .joystick-zone {
    width: 180px;
    height: 180px;
    border-radius: 50%;
    background: rgba(15, 52, 96, 0.5);
    border: 2px solid #0f3460;
    position: relative;
    flex-shrink: 0;
  }

  .center-panel {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 16px;
  }

  .stop-btn {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    background: #e94560;
    border: 3px solid #c81e45;
    color: white;
    font-size: 15px;
    font-weight: 700;
    font-family: inherit;
    letter-spacing: 2px;
    cursor: pointer;
    transition: transform 0.1s;
  }

  .stop-btn:active { transform: scale(0.92); }

  .conn-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    vertical-align: middle;
  }
  .conn-dot.ok { background: #4ecca3; }
  .conn-dot.err { background: #e94560; }

  .label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    opacity: 0.5;
    margin-bottom: 4px;
  }

  @media (max-width: 500px) {
    .joystick-zone { width: 140px; height: 140px; }
    .stop-btn { width: 80px; height: 80px; font-size: 13px; }
  }
</style>
</head>
<body>

<div class="header">
  <h1>LEKIWI TELEOP</h1>
</div>

<div class="status">
  <div><span>vx </span><strong id="vx">0.00</strong></div>
  <div><span>vy </span><strong id="vy">0.00</strong></div>
  <div><span>omega </span><strong id="omega">0.0</strong></div>
  <div><span class="conn-dot err" id="dot"></span><span id="conn">offline</span></div>
</div>

<div class="controls">
  <div>
    <div class="label" style="text-align:center">move</div>
    <div class="joystick-zone" id="move-zone"></div>
  </div>

  <div class="center-panel">
    <button class="stop-btn" id="stop-btn">STOP</button>
  </div>

  <div>
    <div class="label" style="text-align:center">rotate</div>
    <div class="joystick-zone" id="rotate-zone"></div>
  </div>
</div>

<script>
const VX_MAX = """ + str(XY_MAX) + """;
const VY_MAX = """ + str(XY_MAX) + """;
const OMEGA_MAX = """ + str(THETA_MAX) + """;

let vx = 0, vy = 0, omega = 0;
let ws = null;
let sendInterval = null;

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(proto + '://' + location.host + '/ws');

  ws.onopen = () => {
    document.getElementById('dot').className = 'conn-dot ok';
    document.getElementById('conn').textContent = 'connected';
    sendInterval = setInterval(sendCmd, 50);
  };

  ws.onclose = () => {
    document.getElementById('dot').className = 'conn-dot err';
    document.getElementById('conn').textContent = 'offline';
    clearInterval(sendInterval);
    setTimeout(connect, 1000);
  };

  ws.onerror = () => ws.close();
}

function sendCmd() {
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify({vx: vx, vy: vy, omega: omega}));
    document.getElementById('vx').textContent = vx.toFixed(2);
    document.getElementById('vy').textContent = vy.toFixed(2);
    document.getElementById('omega').textContent = omega.toFixed(1);
  }
}

// Move joystick: forward/back = vx, left/right = vy
const moveJoy = nipplejs.create({
  zone: document.getElementById('move-zone'),
  mode: 'static',
  position: {left: '50%', top: '50%'},
  color: '#0f3460',
  size: 120,
});

moveJoy.on('move', (e, d) => {
  const r = Math.min(d.force, 2) / 2;  // normalize 0-1
  const rad = d.angle.radian;
  vx = Math.sin(rad) * r * VX_MAX;
  vy = -Math.cos(rad) * r * VY_MAX;
  // Swap to match robot frame: stick up = forward (vx+), stick right = strafe right (vy-)
  const rawX = Math.sin(rad) * r;
  const rawY = Math.cos(rad) * r;
  vx = rawY * VX_MAX;
  vy = -rawX * VY_MAX;
});

moveJoy.on('end', () => { vx = 0; vy = 0; });

// Rotate joystick: use horizontal movement for rotation
const rotJoy = nipplejs.create({
  zone: document.getElementById('rotate-zone'),
  mode: 'static',
  position: {left: '50%', top: '50%'},
  color: '#0f3460',
  size: 120,
});

rotJoy.on('move', (e, d) => {
  const r = Math.min(d.force, 2) / 2;
  const rad = d.angle.radian;
  // Only use the horizontal component for rotation
  const rawX = Math.cos(rad) * r;
  omega = -rawX * OMEGA_MAX;
});

rotJoy.on('end', () => { omega = 0; });

// Stop button
document.getElementById('stop-btn').addEventListener('click', () => {
  vx = 0; vy = 0; omega = 0;
  sendCmd();
});

// Keyboard fallback
document.addEventListener('keydown', (e) => {
  const step = 0.05;
  const tstep = 15;
  switch(e.key.toLowerCase()) {
    case 'w': vx = Math.min(vx + step, VX_MAX); break;
    case 's': vx = Math.max(vx - step, -VX_MAX); break;
    case 'a': vy = Math.min(vy + step, VY_MAX); break;
    case 'd': vy = Math.max(vy - step, -VY_MAX); break;
    case 'q': omega = Math.min(omega + tstep, OMEGA_MAX); break;
    case 'e': omega = Math.max(omega - tstep, -OMEGA_MAX); break;
    case ' ': vx = 0; vy = 0; omega = 0; e.preventDefault(); break;
  }
});

connect();
</script>
</body>
</html>
"""


# ── FastAPI App ────────────────────────────────────────────────────

app = FastAPI()
bus = None
vel_reg = None


@app.on_event("startup")
async def startup():
    global bus, vel_reg
    try:
        bus = FeetechMotorsBus(port=MOTOR_PORT, motors=WHEELS)
        bus.connect()
        print(f"[motors] Connected on {MOTOR_PORT}")
        bus.disable_torque()
        for name in WHEELS:
            bus.write("Operating_Mode", name, VELOCITY_MODE)
        bus.enable_torque()
        vel_reg = find_velocity_register(bus)
        print(f"[motors] Velocity register: {vel_reg}")
    except Exception as e:
        print(f"[motors] Init failed: {e}")
        print("[motors] Running without motors.")


@app.on_event("shutdown")
async def shutdown():
    if bus and vel_reg:
        try:
            for name in WHEELS:
                bus.write(vel_reg, name, 0)
        except Exception:
            pass
        close_quietly(bus)
    print("[motors] Shut down.")


@app.get("/")
async def index():
    return HTMLResponse(HTML_PAGE)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[ws] Client connected")
    try:
        while True:
            data = await websocket.receive_text()
            cmd = json.loads(data)
            vx = max(-XY_MAX, min(cmd.get("vx", 0), XY_MAX))
            vy = max(-XY_MAX, min(cmd.get("vy", 0), XY_MAX))
            omega = max(-THETA_MAX, min(cmd.get("omega", 0), THETA_MAX))

            raw = body_to_wheel_raw(vx, vy, omega)

            if bus and vel_reg:
                for name, value in zip(WHEELS, raw):
                    bus.write(vel_reg, name, value)

    except WebSocketDisconnect:
        print("[ws] Client disconnected")
        # Stop motors on disconnect
        if bus and vel_reg:
            for name in WHEELS:
                try:
                    bus.write(vel_reg, name, 0)
                except Exception:
                    pass


if __name__ == "__main__":
    print("\n  LeKiwi Web Teleop")
    print("  Open http://<ROBOT_IP>:8000 on your laptop or iPad\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
