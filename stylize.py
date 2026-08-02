#!/usr/bin/env python3
"""Restyle a video into an AI-generated animated version via fal.ai.

Uses Kling O1 video-to-video editing: the whole scene is redrawn per the
prompt while motion and framing are preserved, so the result stays aligned
with the original footage and composite.py can use the finger frame as a
window over it.

Usage:
    export FAL_KEY=...   # https://fal.ai/dashboard/keys
    python stylize.py finger-effect-raw.mov -o stylized.mp4
"""

import argparse
import os
import sys
import urllib.request

MODEL = "fal-ai/kling-video/o1/video-to-video/edit"
DEFAULT_PROMPT = (
    "Transform the person into a 3D animated movie character (stylized CGI "
    "animation look, expressive big eyes, soft lighting). Keep the same "
    "pose, motion, framing, clothing colors, and background composition."
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", nargs="?", default="finger-effect-raw.mov")
    ap.add_argument("-o", "--output", default="stylized.mp4")
    ap.add_argument("-p", "--prompt", default=DEFAULT_PROMPT)
    args = ap.parse_args()

    if not os.environ.get("FAL_KEY"):
        sys.exit("Set FAL_KEY first — create one at https://fal.ai/dashboard/keys")
    if not os.path.exists(args.video):
        sys.exit(f"Input video not found: {args.video}")

    import fal_client  # imported late so --help works without the dependency

    print(f"Uploading {args.video} …")
    video_url = fal_client.upload_file(args.video)

    print(f"Generating with {MODEL} (typically a few minutes) …")

    def on_queue_update(update):
        if isinstance(update, fal_client.InProgress):
            for log in update.logs or []:
                print("   ", log.get("message", ""))

    result = fal_client.subscribe(
        MODEL,
        arguments={"video_url": video_url, "prompt": args.prompt},
        with_logs=True,
        on_queue_update=on_queue_update,
    )

    try:
        out_url = result["video"]["url"]
    except (KeyError, TypeError):
        sys.exit(f"Unexpected API response, inspect it manually:\n{result}")

    print(f"Downloading result to {args.output} …")
    urllib.request.urlretrieve(out_url, args.output)
    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
