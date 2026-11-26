const $ = (selector) => document.querySelector(selector);
const messageInput = $("#message-input");
const subjectInput = $("#subject-input");
const errorMessage = $("#error-message");
const sampleMessages = {
  spam: "Congratulations! You have been selected to receive a FREE entry in our weekly prize draw. Call now to claim your cash reward. Reply STOP to unsubscribe.",
  ham: "Hi, are we still meeting for lunch tomorrow? I can bring the notes from the project discussion. Let me know what time works for you."
};

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.setTimeout(() => toast.classList.remove("show"), 2600);
}

async function api(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "The request could not be completed.");
  return data;
}

function updateCharCount() {
  $("#char-count").textContent = `${messageInput.value.length} / 10,000`;
}

function formatDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}

function renderResult(result) {
  $("#result-placeholder").classList.add("hidden");
  $("#result-content").classList.remove("hidden");
  const isSpam = result.label === "spam";
  const badge = $("#result-badge");
  badge.className = `result-badge${isSpam ? " spam" : ""}`;
  badge.textContent = isSpam ? "!" : "✓";
  $("#result-title").textContent = isSpam ? "Spam detected" : "Looks legitimate";
  $("#result-explanation").textContent = result.explanation;
  $("#score-value").textContent = `${Math.round(result.confidence * 100)}%`;
  const fill = $("#score-fill");
  fill.className = isSpam ? "spam" : "";
  fill.style.width = `${Math.min(100, Math.max(0, result.confidence * 100))}%`;
}

async function analyzeMessage() {
  errorMessage.textContent = "";
  const subject = subjectInput.value.trim();
  const body = messageInput.value.trim();
  if (!subject && !body) {
    errorMessage.textContent = "Enter a message or email to analyze.";
    messageInput.focus();
    return;
  }
  const button = $("#analyze-button");
  button.disabled = true;
  $(".button-label").textContent = "Analyzing…";
  try {
    const result = await api("/api/predict", {
      method: "POST",
      body: JSON.stringify({ subject, body })
    });
    renderResult(result);
    await refreshDashboard();
    showToast("Message analyzed and saved to history.");
  } catch (error) {
    errorMessage.textContent = error.message;
  } finally {
    button.disabled = false;
    $(".button-label").textContent = "Analyze message";
  }
}

function drawDonut(spam, ham) {
  const canvas = $("#donut-chart");
  const context = canvas.getContext("2d");
  const scale = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * scale;
  canvas.height = canvas.clientHeight * scale;
  context.scale(scale, scale);
  const centerX = canvas.clientWidth / 2;
  const centerY = canvas.clientHeight / 2;
  const radius = 43;
  const total = spam + ham;
  context.lineWidth = 10;
  context.lineCap = "round";
  context.beginPath();
  context.strokeStyle = "#eceef4";
  context.arc(centerX, centerY, radius, 0, Math.PI * 2);
  context.stroke();
  if (!total) return;
  let start = -Math.PI / 2;
  for (const [value, color] of [[spam, "#f17878"], [ham, "#7779ed"]]) {
    if (!value) continue;
    const end = start + (value / total) * Math.PI * 2;
    context.beginPath();
    context.strokeStyle = color;
    context.arc(centerX, centerY, radius, start, end);
    context.stroke();
    start = end;
  }
}

function renderBars(spam, ham) {
  const maximum = Math.max(spam, ham, 1);
  $("#bar-chart").innerHTML = [
    { value: spam, label: "Spam", className: "spam" },
    { value: ham, label: "Legitimate", className: "ham" }
  ].map(item => `
    <div class="bar-column">
      <span class="bar-value">${item.value}</span>
      <div class="bar ${item.className}" style="height:${Math.max(3, item.value / maximum * 78)}px"></div>
    </div>`).join("");
}

async function refreshStats() {
  const stats = await api("/api/stats");
  $("#stat-total").textContent = stats.total.toLocaleString();
  $("#stat-spam").textContent = stats.spam.toLocaleString();
  $("#stat-ham").textContent = stats.not_spam.toLocaleString();
  $("#stat-rate").innerHTML = `${(stats.spam_rate * 100).toFixed(1)}<small>%</small>`;
  $("#chart-total").textContent = stats.total;
  $("#legend-spam").textContent = stats.spam;
  $("#legend-ham").textContent = stats.not_spam;
  drawDonut(stats.spam, stats.not_spam);
  renderBars(stats.spam, stats.not_spam);
}

async function refreshHistory() {
  const params = new URLSearchParams();
  const query = $("#history-search").value.trim();
  const label = $("#history-filter").value;
  if (query) params.set("q", query);
  if (label) params.set("label", label);
  const data = await api(`/api/history?${params.toString()}`);
  const body = $("#history-body");
  if (!data.items.length) {
    body.innerHTML = '<tr><td class="empty-state" colspan="4">No matching predictions yet.</td></tr>';
  } else {
    body.innerHTML = data.items.map(item => `
      <tr>
        <td title="${escapeHtml(item.text)}">${escapeHtml(item.text)}</td>
        <td><span class="label-pill ${item.label}">${item.label === "spam" ? "Spam" : "Legitimate"}</span></td>
        <td>${Math.round(item.confidence * 100)}%</td>
        <td>${escapeHtml(formatDate(item.created_at))}</td>
      </tr>`).join("");
  }
  $("#history-count").textContent = `Showing ${data.items.length} prediction${data.items.length === 1 ? "" : "s"}`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[character]);
}

async function refreshDashboard() {
  try {
    await Promise.all([refreshStats(), refreshHistory()]);
  } catch (error) {
    showToast(error.message);
  }
}

async function loadModelInfo() {
  try {
    const info = await api("/api/model-info");
    const modelName = info.model_name || "Trained classifier";
    $("#model-name").textContent = modelName;
    $("#snapshot-model").textContent = modelName;
    const metrics = info.metrics || {};
    $("#metric-precision").textContent = `${((metrics.spam_precision || 0) * 100).toFixed(1)}%`;
    $("#metric-recall").textContent = `${((metrics.spam_recall || 0) * 100).toFixed(1)}%`;
    $("#metric-f1").textContent = `${((metrics.spam_f1 || 0) * 100).toFixed(1)}%`;
  } catch (error) {
    $("#model-name").textContent = "Model unavailable";
    $("#snapshot-model").textContent = "Model unavailable";
    showToast(error.message);
  }
}

$("#analyze-button").addEventListener("click", analyzeMessage);
messageInput.addEventListener("input", updateCharCount);
messageInput.addEventListener("keydown", event => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") analyzeMessage();
});
$("#clear-button").addEventListener("click", () => {
  subjectInput.value = "";
  messageInput.value = "";
  errorMessage.textContent = "";
  updateCharCount();
  messageInput.focus();
});
$("#paste-button").addEventListener("click", async () => {
  try {
    messageInput.value = await navigator.clipboard.readText();
    updateCharCount();
    messageInput.focus();
  } catch {
    showToast("Clipboard access is unavailable. Use Ctrl+V to paste.");
  }
});
document.querySelectorAll(".example-chip").forEach(button => {
  button.addEventListener("click", () => {
    subjectInput.value = button.dataset.example === "spam" ? "You have been selected!" : "Lunch tomorrow";
    messageInput.value = sampleMessages[button.dataset.example];
    updateCharCount();
    messageInput.focus();
  });
});
$("#history-search").addEventListener("input", () => {
  window.clearTimeout(window.historySearchTimer);
  window.historySearchTimer = window.setTimeout(refreshHistory, 200);
});
$("#history-filter").addEventListener("change", refreshHistory);
$("#clear-history-button").addEventListener("click", async () => {
  if (!window.confirm("Clear all saved prediction history? This cannot be undone.")) return;
  try {
    const result = await api("/api/history", { method: "DELETE" });
    await refreshDashboard();
    showToast(`${result.deleted} history record${result.deleted === 1 ? "" : "s"} cleared.`);
  } catch (error) {
    showToast(error.message);
  }
});
$("#theme-toggle").addEventListener("click", () => {
  document.body.classList.toggle("dark");
  localStorage.setItem("inboxguard-theme", document.body.classList.contains("dark") ? "dark" : "light");
  drawDonut(Number($("#legend-spam").textContent), Number($("#legend-ham").textContent));
});
$("#menu-toggle").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
document.querySelectorAll(".nav-link").forEach(link => {
  link.addEventListener("click", () => {
    document.querySelectorAll(".nav-link").forEach(item => item.classList.remove("active"));
    link.classList.add("active");
    $("#sidebar").classList.remove("open");
  });
});
if (localStorage.getItem("inboxguard-theme") === "dark") document.body.classList.add("dark");
updateCharCount();
loadModelInfo();
refreshDashboard();
window.addEventListener("resize", () => drawDonut(Number($("#legend-spam").textContent), Number($("#legend-ham").textContent)));
