# Kairos Action Condition Audit

## Which JSON fields are accepted by `examples/inference.py`?

- `examples/inference.py` accepts any JSON keys syntactically, but after removing `output_dir` and `use_prompt_rewriter` it forwards the remaining keys directly to `pipeline(**input_args_d)`.
- In practice, the safe accepted keys are the pipeline call parameters plus `output_dir` and `use_prompt_rewriter`.

## Which fields reach the Kairos pipeline?

- These action/camera fields reach the pipeline call: `camera_control_direction`, `camera_control_speed`, `camera_control_origin`, `input_image`, `prompt`, `negative_prompt`, `num_frames`, `height`, `width`, `seed`.
- `output_dir` and `use_prompt_rewriter` are consumed by `examples/inference.py` and do not reach the pipeline.

## Which fields are ignored?

- `camera_control_direction`, `camera_control_speed`, and `camera_control_origin` are effectively ignored when `camera_control_direction` is `None`.
- `input_video` is supported by the pipeline itself, but the current JSON wrapper path does not deserialize it into frame lists, so it is not safe to rely on through `examples/inference.py`.

## Which fields cause errors?

- Unknown extra JSON keys cause `TypeError` because `examples/inference.py` forwards them into the strict pipeline call.
- Omitting `prompt` causes `TypeError` because the pipeline signature requires it even when the string is empty.
- Setting `camera_control_direction` without `input_image` is likely to fail because `WanVideoUnit_FunCameraControl` calls `input_image.resize(...)`.

## Camera/action field answers

- `camera_control_direction` works in I2V on this machine: no, the current config/checkpoint path crashes because `pipe.dit.control_adapter` is missing.
- `camera_control_direction` works in TI2V on this machine: no, for the same runtime reason.
- `input_video` changes behavior: yes in the pipeline, and the local wrapper path now supports frame-directory or frame-path-list JSON payloads.
- Valid `camera_control_direction` values: Left, Right, Up, Down, LeftUp, LeftDown, RightUp, RightDown.
- `camera_control_speed` range: no explicit validation in code; default is `1/54`, and this project has successfully used `1.0`.
- `camera_control_origin` matters: yes, it is passed into `process_camera_coordinates(...)` whenever camera control is active.

## Notes

- Audit JSON: `logs/action_conditioning/kairos_action_field_audit.json`
- The working local inference path remains `sensenova_drone_agent/scripts/run_kairos_inference.sh`, not `examples/inference.sh`.
- The current action-conditioning experiment uses an explicit `input_video` fallback whenever the configured Kairos runtime cannot honor `camera_control_direction`.
