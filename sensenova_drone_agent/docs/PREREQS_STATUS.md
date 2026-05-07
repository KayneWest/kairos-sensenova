# Final Prerequisite Status

## Host
- OS: Ubuntu 22.04.5 LTS
- Architecture: x86_64
- Python: Python 3.11.15
- GPU: NVIDIA GPU present
- Existing Kairos repo: /home/mkrzus/kairos-sensenova
- Existing PX4 repo: /home/mkrzus/kairos-sensenova/sensenova_drone_agent/third_party/PX4-Autopilot
- Existing ROS 2: not installed on host
- Existing Gazebo: not installed on host
- Docker: installed and usable without sudo

## Decision
- Chosen ROS distro: jazzy (inside Docker image based on Ubuntu 24.04)
- Chosen Gazebo/PX4 path: Dockerized PX4 SITL source-build image plus Dockerized ROS 2 Jazzy tools image
- Notes: Host package installation was intentionally avoided. ROS 2, Gazebo, bridging, and MAVSDK tooling run inside containers. The final flight-verified path uses a PX4 source-build container layered on `px4io/px4-dev:v1.17.0-rc2`.

## Summary
- PX4 installed: true
- Gazebo installed: true
- ROS 2 installed: true
- ROS/Gazebo bridge installed: true
- MAVSDK-Python installed: true
- QGroundControl installed: false
- Camera frame captured: true
- MAVSDK connection verified: true
- Takeoff/land verified: true
- Offboard nudge verified: true

## Exact commands that worked
- `cd /home/mkrzus/kairos-sensenova/sensenova_drone_agent && ./scripts/install_px4_sitl.sh`
- `cd /home/mkrzus/kairos-sensenova/sensenova_drone_agent && ./scripts/install_ros_gz_bridge.sh`
- `cd /home/mkrzus/kairos-sensenova/sensenova_drone_agent && ./scripts/launch_px4_gazebo_x500_depth.sh --headless --world forest --pose 6,0,1.8,0,0,1.5708`
- `cd /home/mkrzus/kairos-sensenova/sensenova_drone_agent && ./scripts/list_gazebo_topics.sh`
- `cd /home/mkrzus/kairos-sensenova/sensenova_drone_agent && ./scripts/bridge_gazebo_camera_to_ros.sh --topic /world/forest/model/x500_depth_0/link/camera_link/sensor/IMX214/image`
- `cd /home/mkrzus/kairos-sensenova/sensenova_drone_agent && docker compose -f docker-compose.yml run --rm tools bash -lc "set -eo pipefail; set +u; source /opt/ros/\${ROS_DISTRO}/setup.bash; set -u; ros2 run ros_gz_bridge parameter_bridge '/world/forest/model/x500_depth_0/link/camera_link/sensor/IMX214/image@sensor_msgs/msg/Image@gz.msgs.Image' >/tmp/parameter_bridge.log 2>&1 & BRIDGE_PID=\$!; sleep 6; python3 scripts/verify_ros_camera_frame.py --topic '/world/forest/model/x500_depth_0/link/camera_link/sensor/IMX214/image' --out sim_assets/sample_frames/gazebo_rgb_000001.png --timeout 30; kill \${BRIDGE_PID}; wait \${BRIDGE_PID} || true"`
- `cd /home/mkrzus/kairos-sensenova/sensenova_drone_agent && docker compose -f docker-compose.yml run --rm tools python3 scripts/verify_mavsdk_connection.py`
- `cd /home/mkrzus/kairos-sensenova/sensenova_drone_agent && docker compose -f docker-compose.yml run --rm tools python3 scripts/verify_mavsdk_takeoff_hover_land.py --i-understand-this-is-sitl`
- `cd /home/mkrzus/kairos-sensenova/sensenova_drone_agent && docker compose -f docker-compose.yml run --rm tools python3 scripts/verify_mavsdk_offboard_nudge.py --i-understand-this-is-sitl`

## Exact camera topic used
- RGB image topic: `/world/forest/model/x500_depth_0/link/camera_link/sensor/IMX214/image`
- Camera info topic: `/world/forest/model/x500_depth_0/link/camera_link/sensor/IMX214/camera_info`
- Depth topic: `/depth_camera`

## Exact MAVSDK connection string used
- `udpin://0.0.0.0:14540`

## Known issues
- The earlier prebuilt runtime-image path was rejected because PX4 did not publish `sensor_baro` or `sensor_gps` correctly there. The working path is now the source-build sim image documented in `docs/PX4_GAZEBO_SETUP.md`.
- `sim_assets/sample_frames/gazebo_rgb_000001.png` was written from a container and then ownership was reset to the local user.
- The sample RGB frame was refreshed in the `forest` world using `--pose 6,0,1.8,0,0,1.5708`, which yields a reproducible tree-line scene that is more useful for later inference testing than the sparse `default` or `walls` horizon views.
- The `forest` world pulls Fuel assets. It loaded successfully in this environment and produced the current sample frame.
- MAVSDK Offboard verification prints `Received ack for not-existing command: 176! Ignoring...` twice during the run, but the script exits successfully, Offboard starts, the yaw nudge command is accepted, and the vehicle lands cleanly.
- If a user runs the MAVSDK Python scripts directly on the host Python interpreter, they must either activate a MAVSDK-capable environment or use the Docker `tools` service command shown above.
- No Kairos inference integration was attempted in this phase.

## Ready for next phase?
- Yes. The Docker path now proves the simulator camera feed, ROS image bridging, frame capture, MAVSDK connection, simulated takeoff/hover/land, and the tiny Offboard yaw nudge using PX4 SITL only.
- Remaining work belongs to the inference phase, not the prerequisite phase.

READY_FOR_INFERENCE_PHASE=true
