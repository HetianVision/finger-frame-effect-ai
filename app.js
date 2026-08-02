import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const FAL_MODEL = "fal-ai/kling-video/o1/video-to-video/edit";
const STYLES = {
  movie3d:
    "Transform the person into a 3D animated movie character (stylized CGI " +
    "animation look, expressive big eyes, soft lighting).",
  anime:
    "Redraw the video as a hand-drawn anime with clean line art, cel " +
    "shading, and vibrant colors.",
  clay: "Transform the scene into claymation stop-motion with visible clay texture.",
  watercolor:
    "Repaint the video as a soft watercolor painting with loose brushwork.",
};
const PROMPT_SUFFIX =
  " Keep the same pose, motion, framing, clothing colors, and background composition.";

const WRIST = 0, THUMB_TIP = 4, INDEX_TIP = 8, MIDDLE_MCP = 9;

// Tracking constants — same audited pipeline as the live web app.
const MAX_LOST_FRAMES = 25;
const JUMP_CONFIRM_FRAMES = 2;

const orig = document.getElementById("orig");
const sty = document.getElementById("sty");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const stage = document.getElementById("stage");
const drop = document.getElementById("drop");
const btnGenerate = document.getElementById("btn-generate");
const btnPlaceholder = document.getElementById("btn-placeholder");
const btnPlay = document.getElementById("btn-play");
const btnExport = document.getElementById("btn-export");

let landmarker = null;
let videoFile = null;
let haveAI = false;        // stylized video loaded from fal
let usePlaceholder = false; // hue-shift stand-in for keyless testing
let corners = null;
let presence = 0;
let frameActive = false;
let lostFrames = 0;
let jumpFrames = 0;
let recorder = null;
let exporting = false;

function status(msg) {
  statusEl.textContent = msg;
}

// ---- key + style panel ----
const keyInput = document.getElementById("fal-key");
const keyRemember = document.getElementById("fal-remember");
const styleSelect = document.getElementById("style-select");
const styleCustom = document.getElementById("style-custom");

keyInput.value =
  localStorage.getItem("fal-key") || sessionStorage.getItem("fal-key") || "";
keyRemember.checked = !!localStorage.getItem("fal-key");
styleSelect.value = localStorage.getItem("ai-style") || "movie3d";
styleCustom.value = localStorage.getItem("ai-style-custom") || "";
styleCustom.classList.toggle("hidden", styleSelect.value !== "custom");
styleSelect.addEventListener("change", () => {
  styleCustom.classList.toggle("hidden", styleSelect.value !== "custom");
  localStorage.setItem("ai-style", styleSelect.value);
});
styleCustom.addEventListener("change", () =>
  localStorage.setItem("ai-style-custom", styleCustom.value)
);
function saveKey() {
  const key = keyInput.value.trim();
  localStorage.removeItem("fal-key");
  sessionStorage.removeItem("fal-key");
  if (key) (keyRemember.checked ? localStorage : sessionStorage).setItem("fal-key", key);
  return key;
}
function prompt() {
  const style =
    styleSelect.value === "custom" && styleCustom.value.trim()
      ? styleCustom.value.trim()
      : STYLES[styleSelect.value] || STYLES.movie3d;
  return style + PROMPT_SUFFIX;
}

// ---- video loading ----
document.getElementById("file").addEventListener("change", (e) => {
  if (e.target.files[0]) loadVideo(e.target.files[0]);
});
drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  drop.classList.add("over");
});
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  if (e.dataTransfer.files[0]) loadVideo(e.dataTransfer.files[0]);
});

async function loadVideo(file) {
  videoFile = file;
  haveAI = false;
  usePlaceholder = false;
  btnPlay.disabled = true;
  btnExport.disabled = true;
  orig.src = URL.createObjectURL(file);
  await new Promise((res) => (orig.onloadedmetadata = res));
  canvas.width = orig.videoWidth;
  canvas.height = orig.videoHeight;
  stage.style.display = "flex";
  drawPoster();
  status(
    `Loaded ${file.name} (${orig.videoWidth}×${orig.videoHeight}, ` +
    `${orig.duration.toFixed(1)}s). Generate the AI video, or test with the placeholder.`
  );
  if (!landmarker) initLandmarker();
}

function drawPoster() {
  orig.currentTime = 0.01;
  orig.onseeked = () => {
    ctx.drawImage(orig, 0, 0, canvas.width, canvas.height);
    orig.onseeked = null;
  };
}

async function initLandmarker() {
  status("Loading hand tracker…");
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  landmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.3,
    minHandPresenceConfidence: 0.3,
    minTrackingConfidence: 0.3,
  });
  status("Hand tracker ready.");
}

// Dev hook: render one frame at time t (seconds) without realtime playback —
// lets automated tests drive the pipeline in environments that suspend media.
window.__step = (t) =>
  new Promise((resolve) => {
    orig.onseeked = () => {
      orig.onseeked = null;
      ctx.drawImage(orig, 0, 0, canvas.width, canvas.height);
      const res = landmarker.detectForVideo(orig, performance.now());
      updateTracker(res.landmarks || []);
      if (corners && presence > 0.01) {
        drawWindow(corners);
        drawOutline(corners, t);
      }
      resolve({
        presence: +presence.toFixed(2),
        corners: corners
          ? corners.map((p) => [Math.round(p.x), Math.round(p.y)])
          : null,
      });
    };
    orig.currentTime = t;
  });

// Dev convenience: ?src=file.mov loads a local file from the dev server.
const srcParam = new URLSearchParams(location.search).get("src");
if (srcParam) {
  fetch(srcParam)
    .then((r) => r.blob())
    .then((b) => loadVideo(new File([b], srcParam, { type: "video/quicktime" })));
}

// ---- AI generation (fal.ai, BYOK) ----
btnGenerate.addEventListener("click", async () => {
  const key = saveKey();
  if (!key) {
    status("Add your fal.ai key above first (or use the placeholder).");
    keyInput.focus();
    return;
  }
  if (!videoFile) return;
  btnGenerate.disabled = true;
  try {
    status("Loading fal client…");
    const { fal } = await import("https://esm.sh/@fal-ai/client@1");
    fal.config({ credentials: key });
    status("Uploading video to fal.ai…");
    const videoUrl = await fal.storage.upload(videoFile);
    status("Generating — this typically takes a few minutes…");
    const result = await fal.subscribe(FAL_MODEL, {
      input: { video_url: videoUrl, prompt: prompt() },
      logs: true,
      onQueueUpdate(u) {
        if (u.status === "IN_QUEUE") status(`Queued (position ${u.queue_position ?? "…"})…`);
        else if (u.status === "IN_PROGRESS") {
          const last = (u.logs || []).slice(-1)[0];
          status("Generating… " + (last ? last.message : ""));
        }
      },
    });
    const url = result?.data?.video?.url;
    if (!url) throw new Error("no video in response");
    status("Downloading result…");
    const blob = await (await fetch(url)).blob();
    sty.src = URL.createObjectURL(blob);
    await new Promise((res) => (sty.onloadedmetadata = res));
    haveAI = true;
    usePlaceholder = false;
    btnPlay.disabled = false;
    btnExport.disabled = false;
    status("AI video ready — preview or export.");
  } catch (err) {
    console.error(err);
    status("⚠️ Generation failed: " + (err?.body?.detail || err.message || err));
  } finally {
    btnGenerate.disabled = false;
  }
});

btnPlaceholder.addEventListener("click", () => {
  if (!videoFile) return;
  usePlaceholder = true;
  haveAI = false;
  btnPlay.disabled = false;
  btnExport.disabled = false;
  status("Placeholder style active (hue shift) — preview or export, no key needed.");
});

// ---- tracking (ported from finger-frame-effect main.js) ----
function toPixel(lm) {
  return { x: lm.x * canvas.width, y: lm.y * canvas.height };
}
function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}
function lerpPt(a, b, t) {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
}
function polygonArea(pts) {
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i], q = pts[(i + 1) % pts.length];
    a += p.x * q.y - q.x * p.y;
  }
  return Math.abs(a / 2);
}

function computeQuad(hands) {
  if (hands.length !== 2) return null;
  const info = hands.map((lm) => ({
    index: toPixel(lm[INDEX_TIP]),
    thumb: toPixel(lm[THUMB_TIP]),
    wristX: toPixel(lm[WRIST]).x,
    scale: dist(toPixel(lm[WRIST]), toPixel(lm[MIDDLE_MCP])) + 1,
  }));
  const needed = frameActive ? 0.2 : 0.75;
  for (const hd of info) {
    if (dist(hd.thumb, hd.index) < hd.scale * needed) return null;
  }
  info.sort((a, b) => a.wristX - b.wristX);
  const [A, B] = info;
  const pts = [A.index, B.index, B.thumb, A.thumb];
  const cx = pts.reduce((s, p) => s + p.x, 0) / 4;
  const cy = pts.reduce((s, p) => s + p.y, 0) / 4;
  const hull = [...pts].sort(
    (a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx)
  );
  const minArea = frameActive ? 0.0005 : 0.005;
  if (polygonArea(hull) < canvas.width * canvas.height * minArea) return null;
  return pts;
}

function updateTracker(hands) {
  const target = computeQuad(hands);
  if (target) {
    if (!corners) {
      lostFrames = 0;
      frameActive = true;
      jumpFrames = 0;
      corners = target;
      presence = Math.min(1, presence + 0.12);
    } else {
      const moved = target.reduce((s, p, i) => s + dist(p, corners[i]), 0) / 4;
      if (moved > canvas.width * 0.3 && ++jumpFrames < JUMP_CONFIRM_FRAMES) {
        if (++lostFrames > MAX_LOST_FRAMES) presence = Math.max(0, presence - 0.05);
      } else {
        lostFrames = 0;
        frameActive = true;
        jumpFrames = 0;
        const alpha = Math.min(0.85, Math.max(0.35, moved / (canvas.width * 0.05)));
        corners = corners.map((c, i) => lerpPt(c, target[i], alpha));
        presence = Math.min(1, presence + 0.12);
      }
    }
  } else if (corners && ++lostFrames <= MAX_LOST_FRAMES) {
    presence = Math.min(1, presence + 0.12);
  } else {
    presence = Math.max(0, presence - 0.05);
    if (presence === 0) {
      corners = null;
      frameActive = false;
      jumpFrames = 0;
    }
  }
}

// ---- rendering ----
function quadPath(q) {
  ctx.beginPath();
  ctx.moveTo(q[0].x, q[0].y);
  for (let i = 1; i < 4; i++) ctx.lineTo(q[i].x, q[i].y);
  ctx.closePath();
}

function drawWindow(q) {
  ctx.save();
  quadPath(q);
  ctx.clip();
  ctx.globalAlpha = presence;
  if (haveAI) {
    ctx.drawImage(sty, 0, 0, canvas.width, canvas.height);
  } else {
    ctx.filter = "hue-rotate(140deg) saturate(1.7) contrast(1.15)";
    ctx.drawImage(orig, 0, 0, canvas.width, canvas.height);
    ctx.filter = "none";
  }
  ctx.restore();
  ctx.globalAlpha = 1;
}

function drawOutline(q, t) {
  ctx.save();
  ctx.globalAlpha = presence;
  quadPath(q);
  ctx.setLineDash([10, 8]);
  ctx.lineDashOffset = -t * 40;
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(255,255,255,0.95)";
  ctx.shadowColor = "rgba(0,0,0,0.5)";
  ctx.shadowBlur = 6;
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.lineDashOffset = 0;
  ctx.shadowBlur = 0;
  q.forEach((p, i) => {
    const r = 7 + Math.sin(t * 3 + i * 1.5) * 1.5;
    const halo = (t * 0.8 + i * 0.25) % 1;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r + halo * 14, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(255,255,255,${0.5 * (1 - halo) * presence})`;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.fill();
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(0,0,0,0.25)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
  ctx.restore();
}

let lastVideoTime = -1;
function loop() {
  if (!orig.paused && !orig.ended) requestAnimationFrame(loop);

  ctx.drawImage(orig, 0, 0, canvas.width, canvas.height);

  if (landmarker && orig.currentTime !== lastVideoTime) {
    lastVideoTime = orig.currentTime;
    const res = landmarker.detectForVideo(orig, performance.now());
    updateTracker(res.landmarks || []);
  }

  // Keep the stylized video in step with the original.
  if (haveAI && Math.abs(sty.currentTime - orig.currentTime) > 0.15) {
    sty.currentTime = orig.currentTime;
  }

  if (corners && presence > 0.01) {
    drawWindow(corners);
    drawOutline(corners, orig.currentTime);
  }
}

async function playThrough() {
  corners = null;
  presence = 0;
  frameActive = false;
  lostFrames = 0;
  jumpFrames = 0;
  orig.currentTime = 0;
  if (haveAI) {
    sty.currentTime = 0;
    sty.play();
  }
  await orig.play();
  requestAnimationFrame(loop);
}

btnPlay.addEventListener("click", () => {
  if (exporting) return;
  playThrough();
  status("Previewing…");
});

// ---- export (canvas capture -> webm download) ----
btnExport.addEventListener("click", async () => {
  if (exporting) return;
  exporting = true;
  btnExport.disabled = true;
  btnPlay.disabled = true;
  status("Exporting — playing the video through once…");

  const stream = canvas.captureStream(30);
  const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
    ? "video/webm;codecs=vp9"
    : "video/webm";
  recorder = new MediaRecorder(stream, {
    mimeType: mime,
    videoBitsPerSecond: 10_000_000,
  });
  const chunks = [];
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  recorder.onstop = () => {
    const blob = new Blob(chunks, { type: "video/webm" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "finger-frame-ai.webm";
    a.click();
    status(
      "Exported finger-frame-ai.webm. (Convert to mp4 with: " +
      "ffmpeg -i finger-frame-ai.webm -c:v libx264 out.mp4)"
    );
    exporting = false;
    btnExport.disabled = false;
    btnPlay.disabled = false;
  };

  orig.onended = () => {
    orig.onended = null;
    recorder.stop();
  };
  recorder.start();
  await playThrough();
});
