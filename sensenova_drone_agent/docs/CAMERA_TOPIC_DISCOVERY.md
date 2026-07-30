# Camera Topic Discovery

## All Gazebo topics

Saved to `logs/prereqs/gazebo_topics.txt`

## Candidate image/depth topics

Saved to `logs/prereqs/gazebo_camera_topics.txt`

## Chosen RGB topic

/world/forest/model/x500_depth_0/link/camera_link/sensor/IMX214/image

## Chosen depth topic

/depth_camera

## Chosen camera info topic

/world/forest/model/x500_depth_0/link/camera_link/sensor/IMX214/camera_info

## Notes

- Topic metadata saved to `logs/prereqs/gazebo_camera_topic_info.txt`
- Topics were discovered dynamically from the running Gazebo instance on this machine.
- The current sample frame was regenerated in the `forest` world with `--pose 6,0,1.8,0,0,1.5708` so the camera sees a meaningful tree-line scene instead of an empty horizon.
