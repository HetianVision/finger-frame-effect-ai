#!/usr/bin/env python3
"""Restyle a video with Gemini Omni Flash (video-to-video editing).

The whole clip is regenerated per the prompt while motion and framing are
preserved, so the result stays aligned with the original footage and
composite.py can use the finger frame as a window over it.

Usage:
    export GEMINI_API_KEY=...   # https://aistudio.google.com/apikey
    python stylize.py finger-effect-raw.mov -o stylized.mp4

Docs: https://ai.google.dev/gemini-api/docs/omni
"""

import argparse
import os
import sys
import time

MODEL = "gemini-omni-flash-preview"
DEFAULT_PROMPT = (
    "Transform the person into a 3D animated movie character (stylized CGI "
    "animation look, expressive big eyes, soft lighting). Keep the same "
    "pose, motion, framing, clothing colors, and background composition. "
    "Do not change the size or position of the face or body — every feature "
    "must stay exactly where it is in the original."
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", nargs="?", default="finger-effect-raw.mov")
    ap.add_argument("-o", "--output", default="stylized.mp4")
    ap.add_argument("-p", "--prompt", default=DEFAULT_PROMPT)
    args = ap.parse_args()

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        sys.exit("Set GEMINI_API_KEY first — https://aistudio.google.com/apikey")
    if not os.path.exists(args.video):
        sys.exit(f"Input video not found: {args.video}")

    from google import genai  # imported late so --help works without the dep

    client = genai.Client()

    print(f"Uploading {args.video} …")
    video_file = client.files.upload(file=args.video)
    while getattr(video_file, "state", "") and "PROCESS" in str(video_file.state):
        time.sleep(3)
        video_file = client.files.get(name=video_file.name)
    print(f"Uploaded: {video_file.uri}")

    print(f"Generating with {MODEL} (typically a few minutes) …")
    interaction = client.interactions.create(
        model=MODEL,
        input=[
            {"type": "document", "uri": video_file.uri},
            {"type": "text", "text": args.prompt},
        ],
    )

    # Poll if the interaction reports as still running.
    waited = 0
    while (
        str(getattr(interaction, "status", "")).lower()
        in ("pending", "in_progress", "processing", "running", "queued")
        and waited < 900
    ):
        time.sleep(5)
        waited += 5
        interaction = client.interactions.get(id=interaction.id)
        print(f"  … {waited}s ({interaction.status})")

    video_out = getattr(interaction, "output_video", None)
    if video_out is None:
        sys.exit(f"No video in response — raw interaction:\n{interaction}")

    data = getattr(video_out, "data", None)
    if data:
        import base64

        with open(args.output, "wb") as f:
            f.write(base64.b64decode(data) if isinstance(data, str) else data)
    else:
        uri = getattr(video_out, "uri", None)
        if not uri:
            sys.exit(f"No data or uri on output video:\n{video_out}")
        name = "files/" + uri.split("/files/")[1].split("?")[0] if "/files/" in uri else uri
        for _ in range(120):
            f = client.files.get(name=name)
            if "ACTIVE" in str(getattr(f, "state", "")):
                break
            time.sleep(5)
        print("Downloading result …")
        client.files.download(file=f, path=args.output) if hasattr(
            client.files, "download"
        ) else None
        if not os.path.exists(args.output):
            blob = client.files.download(file=f)
            with open(args.output, "wb") as out:
                out.write(blob)

    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
