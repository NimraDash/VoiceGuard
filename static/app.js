// VoiceGuard frontend.
//
// Recording is encoded to WAV in the browser on purpose: it avoids any
// server-side dependency on ffmpeg to decode webm/opus, which is the most
// common way a demo like this breaks on an unfamiliar machine.
//
// Every DOM lookup here is null-safe. A missing element must never be
// reported to the user as a microphone problem.

const $ = (id) => document.getElementById(id);

function setText(id, v) { const e = $(id); if (e) e.textContent = v; }
function setHidden(id, v) { const e = $(id); if (e) e.hidden = v; }
function setStyle(id, prop, v) { const e = $(id); if (e) e.style[prop] = v; }
function setData(id, k, v) { const e = $(id); if (e) e.dataset[k] = v; }

let LOW = 0.3, HIGH = 0.75;
let recorder = null;
let lastUrl = null;

/* ---------- health ---------- */

fetch("/api/health").then(r => r.json()).then(d => {
  LOW = d.thresholds.low;
  HIGH = d.thresholds.high;
  const eer = d.val_metrics && d.val_metrics.eer;
  const bits = [d.model];
  if (typeof eer === "number") bits.push("EER " + (eer * 100).toFixed(1) + "%");
  bits.push(d.policy_loaded ? "policy active" : "thresholds only");
  setText("meta", bits.join("  \u00b7  "));
  paintZones();
}).catch(() => setText("meta", "Model unavailable"));

function paintZones() {
  setStyle("zoneGenuine", "width", (LOW * 100) + "%");
  setStyle("zoneUncertain", "width", ((HIGH - LOW) * 100) + "%");
  setStyle("zoneSynthetic", "width", ((1 - HIGH) * 100) + "%");
  setText("scaleThresholds", LOW.toFixed(2) + " \u2013 " + HIGH.toFixed(2));
}

/* ---------- segmented controls ---------- */

function wireSegmented(id) {
  const root = $(id);
  if (!root) return;
  root.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    [...root.querySelectorAll("button")].forEach(x => x.classList.remove("on"));
    b.classList.add("on");
    root.dataset.value = b.dataset.v;
  });
}
wireSegmented("callerKnown");
wireSegmented("valueBucket");

function ctxValue(id, fallback) {
  const e = $(id);
  return e && e.dataset.value != null ? e.dataset.value : fallback;
}

/* ---------- file upload ---------- */

if ($("file")) {
  $("file").addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setText("filename", f.name);
    if (lastUrl) URL.revokeObjectURL(lastUrl);
    lastUrl = URL.createObjectURL(f);
    const p = $("player");
    if (p) { p.src = lastUrl; setHidden("playback", false); }
    analyze(f);
  });
}

/* ---------- microphone availability ---------- */

function micProblem() {
  // Returns a precise reason string, or null if the mic should be usable.
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    if (!window.isSecureContext) {
      return "Your browser only allows microphone access over https or on "
           + "localhost. You are on " + location.protocol + "//"
           + location.host + ". Upload a file instead, or restart the "
           + "server with:  python app.py --https";
    }
    return "This browser does not expose microphone access. Use Chrome, "
         + "Edge, or Firefox, or upload a file instead.";
  }
  if (!window.AudioContext && !window.webkitAudioContext) {
    return "This browser does not support Web Audio. Upload a file instead.";
  }
  return null;
}

const recBtn = $("recordBtn");
if (recBtn) {
  const initialProblem = micProblem();
  if (initialProblem) {
    recBtn.disabled = true;
    recBtn.title = initialProblem;
  }

  recBtn.addEventListener("click", async () => {
    if (recorder) { stopRecording(); return; }

    const problem = micProblem();
    if (problem) { showError(problem); return; }

    // Step 1: get the microphone. Failures here ARE microphone failures.
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false,
                 autoGainControl: false }
      });
    } catch (err) {
      const n = err && err.name ? err.name : "Error";
      let msg;
      if (n === "NotAllowedError" || n === "SecurityError") {
        msg = "Microphone permission was denied. Click the padlock icon in "
            + "the address bar, allow the microphone, then reload the page.";
      } else if (n === "NotFoundError" || n === "DevicesNotFoundError") {
        msg = "No microphone was found. Plug one in, or upload a file.";
      } else if (n === "NotReadableError" || n === "TrackStartError") {
        msg = "The microphone is in use by another app. Close Zoom, Teams, "
            + "or any recorder, then try again.";
      } else {
        msg = "Microphone error (" + n + "). Upload a file instead.";
      }
      showError(msg);
      return;
    }

    // Step 2: start recording. Failures here are NOT microphone failures
    // and must not be reported as such.
    try {
      await startRecording(stream);
    } catch (err) {
      try { stream.getTracks().forEach(t => t.stop()); } catch (e) {}
      console.error("startRecording failed:", err);
      showError("Recording could not start: "
              + (err && err.message ? err.message : err)
              + ". Try a hard reload (Ctrl+Shift+R) so the page and script "
              + "match, or upload a file instead.");
      resetRecordUi();
    }
  });
}

function resetRecordUi() {
  setText("recordBtn", "Record");
  setData("recordBtn", "recording", "false");
  setHidden("recording", true);
}

/* ---------- recording ---------- */

async function startRecording(stream) {
  const AC = window.AudioContext || window.webkitAudioContext;
  const ctx = new AC();

  // Chrome starts the context suspended until a user gesture resumes it.
  if (ctx.state === "suspended") {
    try { await ctx.resume(); } catch (e) { /* continue anyway */ }
  }

  const source = ctx.createMediaStreamSource(stream);
  const node = ctx.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  let frames = 0;
  let peakSeen = 0;
  let inFlight = false;

  node.onaudioprocess = (e) => {
    const d = e.inputBuffer.getChannelData(0);
    chunks.push(new Float32Array(d));
    frames += d.length;

    let mx = 0;
    for (let i = 0; i < d.length; i += 8) {
      const a = Math.abs(d[i]);
      if (a > mx) mx = a;
    }
    if (mx > peakSeen) peakSeen = mx;
    setStyle("levelFill", "width", Math.min(100, mx * 320) + "%");
  };

  source.connect(node);
  // Route through a silent gain node instead of straight to the speakers,
  // so the microphone is not echoed back into the room during a live demo.
  const sink = ctx.createGain();
  sink.gain.value = 0;
  node.connect(sink);
  sink.connect(ctx.destination);

  const started = Date.now();
  const rate = ctx.sampleRate;

  const timer = setInterval(() => {
    const s = Math.floor((Date.now() - started) / 1000);
    setText("recTime",
            Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0"));
  }, 200);

  const scoreTimer = setInterval(async () => {
    if (inFlight) return;
    if (frames < rate * 1.5) return;

    const need = rate * 4;
    let total = 0;
    chunks.forEach(c => total += c.length);
    const take = Math.min(total, need);
    const win = new Float32Array(take);
    let filled = take;
    for (let i = chunks.length - 1; i >= 0 && filled > 0; i--) {
      const c = chunks[i];
      const n = Math.min(c.length, filled);
      win.set(c.subarray(c.length - n), filled - n);
      filled -= n;
    }

    inFlight = true;
    try {
      const fd = new FormData();
      fd.append("audio", encodeWav(win, rate), "w.wav");
      const res = await fetch("/api/score_window", { method: "POST", body: fd });
      if (res.ok) {
        const d = await res.json();
        setHidden("live", false);
        setData("live", "band", d.band);
        setText("liveScore", d.p.toFixed(2));
        setStyle("liveMarker", "left", (d.p * 100) + "%");
        setText("liveNote", {
          GENUINE: "Sounds like a real voice so far.",
          UNCERTAIN: "Not confident either way. Keep recording.",
          SYNTHETIC: "Signs of an AI-generated voice."
        }[d.band] || "");
      }
    } catch (e) { /* the live view is best-effort */ }
    inFlight = false;
  }, 2000);

  recorder = { ctx, stream, source, node, sink, chunks, timer, scoreTimer,
               rate,
               get frames() { return frames; },
               get peak() { return peakSeen; } };

  setText("recordBtn", "Stop");
  setData("recordBtn", "recording", "true");
  setHidden("recording", false);
  setHidden("live", true);
  setHidden("playback", true);
  setText("recTime", "0:00");
  setStyle("levelFill", "width", "0%");
  setState("idle");
}

function stopRecording() {
  const r = recorder;
  recorder = null;
  if (!r) return;

  try { clearInterval(r.timer); } catch (e) {}
  try { clearInterval(r.scoreTimer); } catch (e) {}
  try { r.node.disconnect(); } catch (e) {}
  try { r.source.disconnect(); } catch (e) {}
  try { r.sink.disconnect(); } catch (e) {}
  try { r.stream.getTracks().forEach(t => t.stop()); } catch (e) {}

  const rate = r.rate;
  const peak = r.peak;
  try { r.ctx.close(); } catch (e) {}

  resetRecordUi();

  let total = 0;
  r.chunks.forEach(c => total += c.length);

  if (total === 0) {
    showError("No audio was captured at all. The microphone may be muted at "
            + "the operating-system level. Check Windows sound settings, or "
            + "upload a file instead.");
    return;
  }

  const merged = new Float32Array(total);
  let off = 0;
  r.chunks.forEach(c => { merged.set(c, off); off += c.length; });
  const blob = encodeWav(merged, rate);

  // Always let them hear what was actually captured.
  if (lastUrl) URL.revokeObjectURL(lastUrl);
  lastUrl = URL.createObjectURL(blob);
  const p = $("player");
  if (p) { p.src = lastUrl; setHidden("playback", false); }

  if (total < rate * 0.8) {
    showError("That clip was only " + (total / rate).toFixed(1) + "s long. "
            + "Record at least one second. You can play it back below.");
    return;
  }

  if (peak < 0.005) {
    showError("The microphone captured almost no sound (peak level "
            + peak.toFixed(4) + "). Play the clip below to confirm, then "
            + "raise your input volume in Windows sound settings.");
    return;
  }

  setText("filename", "Recorded clip (" + (total / rate).toFixed(1) + "s)");
  analyze(blob);
}

function encodeWav(samples, sampleRate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
  };
  str(0, "RIFF");
  v.setUint32(4, 36 + samples.length * 2, true);
  str(8, "WAVE"); str(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  str(36, "data");
  v.setUint32(40, samples.length * 2, true);
  let o = 44;
  for (let i = 0; i < samples.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

/* ---------- analyse ---------- */

function setState(which) {
  setHidden("stateIdle", which !== "idle");
  setHidden("stateBusy", which !== "busy");
  setHidden("stateError", which !== "error");
  setHidden("results", which !== "done");
}

function showError(msg) {
  setText("errorText", msg);
  setState("error");
}

async function analyze(fileOrBlob) {
  setState("busy");
  const fd = new FormData();
  fd.append("audio", fileOrBlob, "clip.wav");
  fd.append("caller_known", ctxValue("callerKnown", "0"));
  fd.append("value_bucket", ctxValue("valueBucket", "0"));
  try {
    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    let data;
    try {
      data = await res.json();
    } catch (e) {
      showError("The server returned an unreadable response. Check the "
              + "server console window.");
      return;
    }
    if (!res.ok) { showError(data.error || "Analysis failed."); return; }
    render(data);
  } catch (err) {
    showError("Could not reach the server. Is it still running?");
  }
}

function bandOf(p) {
  if (p < LOW) return "GENUINE";
  if (p > HIGH) return "SYNTHETIC";
  return "UNCERTAIN";
}

const BAND_LABEL = {
  GENUINE: "Voice appears genuine",
  UNCERTAIN: "Cannot confirm this voice",
  SYNTHETIC: "Likely synthetic voice"
};

function render(d) {
  LOW = d.thresholds.low; HIGH = d.thresholds.high;
  paintZones();

  const r = $("results");
  if (r) { r.dataset.band = d.band; r.dataset.action = d.action; }

  setText("verdictBand", BAND_LABEL[d.band]);
  setText("verdictScore", d.probability.toFixed(2));
  setText("verdictMessage", d.message);
  setStyle("marker", "left", (d.probability * 100) + "%");

  const n = d.windows.length;
  setText("segmentSub",
          n + (n === 1 ? " window" : " windows") + " across "
          + d.duration.toFixed(1) + "s, four seconds each");

  const seg = $("segments");
  if (seg) {
    seg.innerHTML = "";
    d.windows.forEach(w => {
      const el = document.createElement("div");
      el.className = "seg";
      el.dataset.band = bandOf(w.p);
      el.style.height = Math.max(4, w.p * 68) + "px";
      el.title = w.t.toFixed(1) + "s \u2014 " + w.p.toFixed(2);
      seg.appendChild(el);
    });
  }

  setText("actionTitle", d.action_title);
  setText("actionDetail", d.action_detail);
  setText("actionSource", d.action_source === "policy"
    ? "Chosen by the cost-sensitive escalation policy, using the voice score, its trend across windows, and the call context above."
    : "Chosen by threshold rules (escalation policy not applied to this state).");

  setState("done");
}

setState("idle");
