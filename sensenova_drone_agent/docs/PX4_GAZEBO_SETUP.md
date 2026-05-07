# PX4/Gazebo Setup

## Install command used

`docker pull px4io/px4-dev:v1.17.0-rc2`

`docker compose -f /home/mkrzus/kairos-sensenova/sensenova_drone_agent/docker-compose.yml build sim`

## PX4 commit

`f63b0d6b6f1ac8da180292db438b63e7f7b39048`

## Gazebo version

Gazebo Sim, version 8.11.0

## Notes

- Docker sim image: `sensenova_drone_agent-px4-source-sim:local`
- Docker sim image id: `sha256:5fc2a5d6a474aeeed1feb97edf363dd101d3711fa7b6a30e566b20da712acb89`
- Docker dev base image: `px4io/px4-dev:v1.17.0-rc2`
- Docker dev base digest: `px4io/px4-dev@sha256:5e7ad18c75c3a5a655d5adfde4ab1eb216dd4bee7710941b6cd122f3969a7fed`
- Container base OS: Ubuntu 24.04.4 LTS
- Full log: `logs/prereqs/px4_install.log`
- PX4 source is kept in `third_party/PX4-Autopilot` and is built inside the Docker sim container to avoid host package installs.
- This source-build container replaced the earlier prebuilt runtime image path because the prebuilt image did not publish `sensor_baro` and `sensor_gps` correctly in this environment.
- Flight verification now passes with this source-build image: MAVSDK connection, takeoff/hover/land, and the tiny Offboard yaw nudge all succeeded against PX4 SITL.

## Reboot required?

No. The Docker-based path does not modify host PX4/Gazebo packages.
