const connectBtn = document.getElementById("connectBtn");
const disconnectBtn = document.getElementById("disconnectBtn");
const interruptBtn = document.getElementById("interruptBtn");
const userIdInput = document.getElementById("userId");
const phaseEl = document.getElementById("phase");
const sessionIdEl = document.getElementById("sessionId");
const traceIdEl = document.getElementById("traceId");
const transcriptStatusEl = document.getElementById("transcriptStatus");
const playbackStatusEl = document.getElementById("playbackStatus");

const transcriptLog = document.getElementById("transcriptLog");
const agentLog = document.getElementById("agentLog");
const playbackLog = document.getElementById("playbackLog");
const wireLog = document.getElementById("wireLog");

let socket = null;
let audioContext = null;
let mediaStream = null;
let workletNode = null;
let sourceNode = null;
let currentAudio = null;
let audioQueue = [];
let closed = false;
let expectedGeneration = null;
let finalSegmentCount = 0;
let completedSegments = new Set();
let generationDone = false;

function appendLog(target, title, body) {
  const item = document.createElement("div");
  item.className = "log-item";
  item.innerHTML = `<strong>${title}</strong><div>${body}</div>`;
  target.prepend(item);
}

function setButtons(connected) {
  connectBtn.disabled = connected;
  disconnectBtn.disabled = !connected;
  interruptBtn.disabled = !connected;
}

function toBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function fromBase64(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

async function connect() {
  if (socket) {
    return;
  }
  closed = false;
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const userId = encodeURIComponent(userIdInput.value.trim() || "demo-user");
  socket = new WebSocket(`${protocol}://${location.host}/ws/voice?user_id=${userId}`);
  socket.onmessage = handleSocketMessage;
  socket.onclose = () => teardown("socket_closed");
  socket.onerror = () => appendLog(wireLog, "socket", "error");
  await initAudio();
  setButtons(true);
}

async function initAudio() {
  audioContext = new AudioContext();
  await audioContext.audioWorklet.addModule("/static/js/pcm-processor.js");
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });
  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(audioContext, "pcm-chunk-processor", {
    processorOptions: {
      targetSampleRate: 24000,
      chunkFrames: 6000,
    },
  });
  workletNode.port.onmessage = (event) => {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return;
    }
    const payload = {
      type: "audio.append",
      audio: toBase64(event.data.pcm),
      rms: event.data.rms,
    };
    socket.send(JSON.stringify(payload));
  };
  sourceNode.connect(workletNode);
  workletNode.connect(audioContext.destination);
}

function handleSocketMessage(event) {
  const message = JSON.parse(event.data);
  appendLog(wireLog, message.type, JSON.stringify(message, null, 2));

  switch (message.type) {
    case "session.started":
      sessionIdEl.textContent = message.session_id;
      traceIdEl.textContent = message.trace_id;
      phaseEl.textContent = message.phase;
      appendLog(transcriptLog, "session", `started for ${message.user_id}`);
      break;
    case "state":
      phaseEl.textContent = message.phase;
      transcriptStatusEl.textContent = message.phase;
      if (message.phase === "speaking") {
        playbackStatusEl.textContent = "playing";
      }
      if (message.phase === "listening") {
        playbackStatusEl.textContent = "stopped";
      }
      break;
    case "transcript.final":
      appendLog(transcriptLog, "user", message.text || "(empty transcript)");
      break;
    case "handoff":
      appendLog(agentLog, "handoff", `${message.source_agent} -> ${message.target_agent}<br>${message.current_intent}`);
      break;
    case "agent.event":
      appendLog(agentLog, message.agent, `${message.action} | ${message.detail} | ${message.latency_ms} ms`);
      break;
    case "worker.result":
      appendLog(agentLog, `${message.agent} worker`, `success=${message.success} confidence=${message.confidence}`);
      break;
    case "consensus":
      appendLog(agentLog, "consensus", `${message.answer_context}<br>sources=${message.winning_sources.join(", ")}`);
      break;
    case "assistant.reply":
      appendLog(transcriptLog, "assistant", message.text);
      expectedGeneration = message.generation_id;
      finalSegmentCount = message.segment_count;
      completedSegments = new Set();
      generationDone = false;
      break;
    case "audio.segment":
      queueAudioSegment(message);
      break;
    case "audio.done":
      generationDone = true;
      finalSegmentCount = message.segment_count;
      maybeCompleteGeneration();
      break;
    case "audio.stop":
      stopPlayback();
      appendLog(playbackLog, "interrupt", message.reason || "server_stop");
      break;
    case "error":
      appendLog(agentLog, "error", message.message || "unknown");
      break;
    default:
      break;
  }
}

function queueAudioSegment(message) {
  const bytes = fromBase64(message.audio);
  const blob = new Blob([bytes], { type: message.mime_type });
  const url = URL.createObjectURL(blob);
  audioQueue.push({
    url,
    generationId: message.generation_id,
    segmentIndex: message.segment_index,
    segmentCount: message.segment_count,
    text: message.text,
  });
  appendLog(playbackLog, `segment ${message.segment_index + 1}`, message.text);
  if (!currentAudio) {
    playNextSegment();
  }
}

function playNextSegment() {
  if (currentAudio || audioQueue.length === 0) {
    maybeCompleteGeneration();
    return;
  }
  const next = audioQueue.shift();
  currentAudio = new Audio(next.url);
  playbackStatusEl.textContent = `segment ${next.segmentIndex + 1}/${next.segmentCount}`;
  currentAudio.onended = () => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(
        JSON.stringify({
          type: "playback.segment_completed",
          generation_id: next.generationId,
          segment_index: next.segmentIndex,
        }),
      );
    }
    completedSegments.add(next.segmentIndex);
    URL.revokeObjectURL(next.url);
    currentAudio = null;
    playNextSegment();
  };
  currentAudio.play().catch((error) => {
    appendLog(playbackLog, "playback_error", error.message);
    URL.revokeObjectURL(next.url);
    currentAudio = null;
  });
}

function maybeCompleteGeneration() {
  if (!generationDone || expectedGeneration == null) {
    return;
  }
  if (currentAudio || audioQueue.length > 0) {
    return;
  }
  if (completedSegments.size < finalSegmentCount) {
    return;
  }
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(
      JSON.stringify({
        type: "playback.generation_completed",
        generation_id: expectedGeneration,
      }),
    );
  }
  playbackStatusEl.textContent = "stopped";
}

function stopPlayback() {
  audioQueue.forEach((item) => URL.revokeObjectURL(item.url));
  audioQueue = [];
  if (currentAudio) {
    currentAudio.pause();
    URL.revokeObjectURL(currentAudio.src);
    currentAudio = null;
  }
  completedSegments = new Set();
  generationDone = false;
  playbackStatusEl.textContent = "stopped";
}

async function teardown(reason = "manual") {
  if (closed) {
    return;
  }
  closed = true;
  stopPlayback();
  if (workletNode) {
    workletNode.port.onmessage = null;
    workletNode.disconnect();
    workletNode = null;
  }
  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  if (audioContext) {
    await audioContext.close();
    audioContext = null;
  }
  if (socket) {
    socket.onclose = null;
    socket.close();
    socket = null;
  }
  setButtons(false);
  phaseEl.textContent = reason;
}

connectBtn.addEventListener("click", connect);
disconnectBtn.addEventListener("click", () => teardown("manual_disconnect"));
interruptBtn.addEventListener("click", () => {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "interrupt" }));
  }
});
