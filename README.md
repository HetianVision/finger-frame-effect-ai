# Finger Frame AI 🎬✨

Upload a video of the two-hand finger-frame gesture — get it back with an
**AI-generated world inside the frame**. The whole video is restyled by a
video-to-video model (motion, blinks, and all), then composited so the
finger frame acts as a window into the animated version.

Companion to the live-camera app
[finger-frame-effect](https://github.com/sophiamyang/finger-frame-effect).

![Example: AI-animated world inside the finger frame](examples/final.gif)

*Real hands, AI world — generated with the default "3D animated movie"
style ([full-quality mp4](examples/final.mp4)).*

## How it works

1. **Restyle** — the uploaded video is sent to
   [Gemini Omni Flash video editing](https://ai.google.dev/gemini-api/docs/omni)
   with your chosen style (3D animated movie, anime, claymation,
   watercolor, or a custom prompt). This is a true video model: the whole
   clip is regenerated, so the animated version moves exactly like you.
2. **Track** — MediaPipe Hand Landmarker finds both hands per frame, and the
   finger-frame quad is tracked with the same audited pipeline as the live
   app (anatomical corner ordering — crossing your fingers renders the
   bowtie — spread/area gates with hysteresis, teleport rejection,
   velocity-adaptive smoothing, dropout hold, presence fade).
3. **Composite** — the AI video is revealed through the tracked quad with
   the dashed marching-ants outline and pulsing corner dots.
4. **Export** — the result records to a downloadable `.webm`
   (convert with `ffmpeg -i finger-frame-ai.webm -c:v libx264 out.mp4`).

## Bring your own key

The AI step uses your own [Gemini API key](https://aistudio.google.com/apikey),
entered in the app. It stays in your browser (localStorage only if you check
"remember") and is sent only to Google's API. Generation is billed per
video. No key? The **placeholder style** button runs the full
track-composite-export pipeline with a hue-shifted stand-in so you can try
everything for free.

## Run locally

Any static server works:

```bash
python3 -m http.server 8124
```

Then open http://localhost:8124. A `?src=<file>` query param loads a video
from the server directory (dev convenience).

## CLI alternative (Python)

The same pipeline as offline scripts — useful for batch work or
frame-accurate H.264 output:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

export GEMINI_API_KEY=...
.venv/bin/python stylize.py input.mov -o stylized.mp4      # AI restyle
.venv/bin/python composite.py input.mov stylized.mp4 -o final.mp4
```

`composite.py` needs `ffmpeg` on PATH and carries over the original audio
track when present.

## Notes

- Media files are gitignored on purpose — personal footage stays local;
  only the pipeline code is published.
