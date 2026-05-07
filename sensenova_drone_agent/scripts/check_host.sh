#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/prereqs"
DOC_PATH="${PROJECT_ROOT}/docs/PREREQS_STATUS.md"
HOST_AUDIT_PATH="${LOG_DIR}/host_audit.txt"

mkdir -p "${LOG_DIR}"

os_name="Unknown"
os_version="Unknown"
os_pretty="Unknown"
ubuntu_version="unsupported"
architecture="$(uname -m)"
python_version="$(python3 --version 2>/dev/null || echo 'python3 not found')"
gpu_status="No NVIDIA GPU detected"
ros_status="not installed"
gazebo_status="not installed"
px4_repo="not found"
qgc_status="not found"
kairos_repo="not found"

if [[ -f /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  os_name="${NAME:-Unknown}"
  os_version="${VERSION_ID:-Unknown}"
  os_pretty="${PRETTY_NAME:-Unknown}"
  ubuntu_version="${VERSION_ID:-unsupported}"
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  gpu_status="NVIDIA GPU present"
fi

if [[ -d /opt/ros/jazzy ]]; then
  ros_status="installed (/opt/ros/jazzy)"
elif [[ -d /opt/ros/humble ]]; then
  ros_status="installed (/opt/ros/humble)"
elif command -v ros2 >/dev/null 2>&1; then
  ros_status="installed ($(command -v ros2))"
fi

if command -v gz >/dev/null 2>&1; then
  gazebo_status="installed ($(gz sim --version 2>/dev/null | head -n 1 || echo 'version unavailable'))"
elif dpkg -l 2>/dev/null | grep -q '^ii  gz-harmonic '; then
  gazebo_status="installed (gz-harmonic package present)"
fi

if [[ -d "${PROJECT_ROOT}/third_party/PX4-Autopilot" ]]; then
  px4_repo="${PROJECT_ROOT}/third_party/PX4-Autopilot"
else
  px4_repo_candidate="$(find "${PROJECT_ROOT}/.." -maxdepth 3 -type d -name PX4-Autopilot 2>/dev/null | head -n 1 || true)"
  if [[ -n "${px4_repo_candidate}" ]]; then
    px4_repo="${px4_repo_candidate}"
  fi
fi

qgc_candidate="$(find "${PROJECT_ROOT}/.." -maxdepth 4 \( -iname 'QGroundControl*.AppImage' -o -iname 'QGroundControl' \) 2>/dev/null | head -n 1 || true)"
if [[ -n "${qgc_candidate}" ]]; then
  qgc_status="${qgc_candidate}"
elif command -v QGroundControl >/dev/null 2>&1; then
  qgc_status="$(command -v QGroundControl)"
fi

kairos_candidate="$(find "${PROJECT_ROOT}/.." -maxdepth 2 -type d -iname 'kairos-sensenova' 2>/dev/null | head -n 1 || true)"
if [[ -n "${kairos_candidate}" ]]; then
  kairos_repo="${kairos_candidate}"
elif [[ -d "/home/mkrzus/kairos-sensenova" ]]; then
  kairos_repo="/home/mkrzus/kairos-sensenova"
fi

case "${ubuntu_version}" in
  24.04)
    ros_distro="jazzy"
    chosen_path="PX4 SITL + Gazebo Harmonic + ROS 2 Jazzy"
    notes="Preferred target for this host class."
    ;;
  22.04)
    ros_distro="humble"
    chosen_path="PX4 SITL + Gazebo Harmonic + ROS 2 Humble via ros-humble-ros-gzharmonic"
    notes="Acceptable path. Humble plus Harmonic should use ros-humble-ros-gzharmonic rather than a default Humble Gazebo pairing."
    ;;
  *)
    ros_distro="unsupported"
    chosen_path="unsupported"
    notes="Host is not Ubuntu 22.04 or 24.04. Stop before custom OS setup."
    ;;
esac

{
  echo "date"
  date
  echo
  echo "uname -a"
  uname -a
  echo
  echo "lsb_release -a || cat /etc/os-release"
  lsb_release -a || cat /etc/os-release
  echo
  echo "python3 --version"
  python3 --version
  echo
  echo "pip3 --version || true"
  pip3 --version || true
  echo
  echo "git --version"
  git --version
  echo
  echo "cmake --version || true"
  cmake --version || true
  echo
  echo "ninja --version || true"
  ninja --version || true
  echo
  echo "gz sim --version || true"
  gz sim --version || true
  echo
  echo "ros2 --version || true"
  ros2 --version || true
  echo
  echo "nvidia-smi || true"
  nvidia-smi || true
  echo
  echo "Detected summary"
  echo "OS: ${os_pretty}"
  echo "Architecture: ${architecture}"
  echo "GPU: ${gpu_status}"
  echo "Existing Kairos repo: ${kairos_repo}"
  echo "Existing PX4 repo: ${px4_repo}"
  echo "Existing ROS 2: ${ros_status}"
  echo "Existing Gazebo: ${gazebo_status}"
  echo "QGroundControl: ${qgc_status}"
  echo "Chosen ROS distro: ${ros_distro}"
  echo "Chosen Gazebo/PX4 path: ${chosen_path}"
  echo "Notes: ${notes}"
} | tee "${HOST_AUDIT_PATH}"

cat > "${DOC_PATH}" <<EOF
# Prerequisites Status

## Host
- OS: ${os_pretty}
- Architecture: ${architecture}
- Python: ${python_version}
- GPU: ${gpu_status}
- Existing Kairos repo: ${kairos_repo}
- Existing PX4 repo: ${px4_repo}
- Existing ROS 2: ${ros_status}
- Existing Gazebo: ${gazebo_status}

## Decision
- Chosen ROS distro: ${ros_distro}
- Chosen Gazebo/PX4 path: ${chosen_path}
- Notes: ${notes}
EOF

echo "Host audit written to ${HOST_AUDIT_PATH}"
echo "Status document updated at ${DOC_PATH}"
