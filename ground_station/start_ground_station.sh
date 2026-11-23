#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/../../.." 2>/dev/null && pwd || echo "$HOME/ros2_ws")"
export PATH="$HOME/.local/node/bin:$PATH"

# Source ROS 2 Jazzy + workspace overlay if available
if [ -f /opt/ros/jazzy/setup.bash ]; then
  source /opt/ros/jazzy/setup.bash
fi
if [ -f "$WS_DIR/install/setup.bash" ]; then
  source "$WS_DIR/install/setup.bash"
fi

PYTHON_BIN="$WS_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

# Auto-install frontend packages on first run if node_modules is missing
if [ ! -d "$SCRIPT_DIR/frontend/node_modules" ]; then
  echo "Installing frontend dependencies..."
  (cd "$SCRIPT_DIR/frontend" && npm install)
fi

echo "Starting AgriRover Ground Station..."
echo "  - Backend API + GStreamer 1.24 + SQLite: http://localhost:8000"
echo "  - Frontend React UI (Vite Dev):          http://localhost:5173"

cd "$SCRIPT_DIR/backend"
"$PYTHON_BIN" -m uvicorn main:app --host 0.0.0.0 --port 8000 &
BACK_PID=$!

cd "$SCRIPT_DIR/frontend"
npm run dev &
FRONT_PID=$!

trap "kill $BACK_PID $FRONT_PID 2>/dev/null || true" EXIT INT TERM
wait
