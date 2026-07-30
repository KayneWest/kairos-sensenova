from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
from PIL import Image, ImageDraw


def make_video_contact_sheet(
    video_paths: dict[str, str],
    out_path: str,
    num_frames: int = 6,
):
    labels = list(video_paths.keys())
    paths = [Path(video_paths[label]) for label in labels]
    row_images: list[list[Image.Image]] = []
    thumb_width = 320
    thumb_height = 180
    label_width = 220
    row_height = thumb_height + 10

    for path in paths:
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            base = Image.open(path).convert("RGB")
            frames = [base] * num_frames
        else:
            video = iio.imread(path)
            if video.ndim == 3:
                video = video[None, ...]
            total = len(video)
            indices = [min(int(round(i * (total - 1) / max(num_frames - 1, 1))), total - 1) for i in range(num_frames)]
            frames = [Image.fromarray(video[idx]).convert("RGB") for idx in indices]
        row_images.append([frame.resize((thumb_width, thumb_height)) for frame in frames])

    canvas_width = label_width + thumb_width * num_frames
    canvas_height = row_height * len(labels)
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)

    for row_index, (label, frames) in enumerate(zip(labels, row_images)):
        top = row_index * row_height
        draw.rectangle((0, top, canvas_width, top + row_height), fill=(28, 28, 28) if row_index % 2 == 0 else (36, 36, 36))
        draw.text((12, top + 12), label, fill=(255, 255, 255))
        for col_index, frame in enumerate(frames):
            left = label_width + col_index * thumb_width
            canvas.paste(frame, (left, top + 5))
            draw.text((left + 10, top + 12), f"f{col_index + 1}", fill=(255, 255, 255))

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_file)
    return str(out_file)
