const $ = (selector, root = document) => root.querySelector(selector);
const csrfToken = $('meta[name="csrf-token"]')?.content || "";
const themeKey = "inboxguard-theme";
const accountToken = $("#account-token");
if (accountToken && window.location.hash.length > 1) {
  accountToken.value = window.location.hash.slice(1);
  window.history.replaceState(null, "", window.location.pathname);
}

function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[character]);
}

function toast(message) {
  const element = $("#toast");
  if (!element) return;
  element.textContent = message;
  element.classList.add("show");
  window.setTimeout(() => element.classList.remove("show"), 2800);
}

async function requestJSON(url, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (options.body && !(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }
  if (!["GET", "HEAD"].includes((options.method || "GET").toUpperCase())) {
    headers["X-CSRF-Token"] = csrfToken;
  }
  const response = await fetch(url, { ...options, headers, credentials: "same-origin" });
  const type = response.headers.get("content-type") || "";
  const payload = type.includes("application/json") ? await response.json() : {};
  if (!response.ok) {
    if (response.status === 401) window.location.assign("/login");
    throw new Error(payload.error || `Request failed (${response.status}).`);
  }
  return payload;
}

function formatDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? String(value || "")
    : date.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}

function classificationLabel(value) {
  return ({
    safe: "Safe / legitimate",
    promotional: "Promotional",
    spam: "Spam",
    phishing_suspicious: "Phishing / suspicious",
    uncertain: "Uncertain / review"
  })[value] || value || "Unknown";
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem(themeKey, theme);
}

const storedTheme = localStorage.getItem(themeKey);
if (storedTheme === "dark" || storedTheme === "light") setTheme(storedTheme);
$("#theme-toggle")?.addEventListener("click", () => {
  setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
});
$("#menu-toggle")?.addEventListener("click", () => $("#sidebar")?.classList.toggle("open"));

async function loadDashboard() {
  if (!$("#distribution")) return;
  try {
    const stats = await requestJSON("/api/stats");
    $("#stat-total").textContent = stats.total.toLocaleString();
    $("#stat-spam").textContent = stats.spam.toLocaleString();
    $("#stat-legitimate").textContent = stats.legitimate.toLocaleString();
    $("#stat-phishing").textContent = stats.phishing.toLocaleString();
    $("#stat-rate").textContent = `${(stats.spam_rate * 100).toFixed(1)}%`;
    const campaignAlerts = Array.isArray(stats.campaign_alerts) ? stats.campaign_alerts : [];
    const campaignCount = $("#campaign-alert-count");
    const campaignList = $("#campaign-alert-list");
    if (campaignCount && campaignList) {
      campaignCount.textContent = campaignAlerts.length
        ? `${campaignAlerts.length} pattern${campaignAlerts.length === 1 ? "" : "s"} found`
        : "No patterns found";
      campaignList.innerHTML = campaignAlerts.length
        ? campaignAlerts.map(alert => `
          <article class="distribution-row">
            <span><b>${escapeHTML(alert.type === "repeated_sender_domain" ? "Repeated sender domain" : "Repeated subject")}</b><small>${escapeHTML(alert.indicator)}</small></span>
            <b>${escapeHTML(alert.count)} matches</b>
            <p>${escapeHTML(alert.message)}</p>
          </article>`).join("")
        : '<p class="empty-row">No repeated suspicious campaigns were detected in your recent scans.</p>';
    }
    const labels = [
      ["safe", "Safe / legitimate", "#16a34a"],
      ["promotional", "Promotional", "#6366f1"],
      ["spam", "Spam", "#dc2626"],
      ["phishing_suspicious", "Phishing / suspicious", "#d97706"],
      ["uncertain", "Uncertain", "#64748b"]
    ];
    const total = Math.max(stats.total, 1);
    $("#distribution").innerHTML = labels.map(([key, label, color]) => {
      const count = stats.distribution[key] || 0;
      const width = Math.round(count / total * 100);
      return `<div class="distribution-row"><span>${label}</span><b>${count}</b><div><i style="width:${width}%;background:${color}"></i></div></div>`;
    }).join("");
    const trend = stats.trends || [];
    const maxDaily = Math.max(1, ...trend.map(day => day.total));
    $("#activity-chart").innerHTML = trend.length ? trend.map(day => {
      const spamHeight = Math.max(0, day.spam / maxDaily * 78);
      const otherHeight = Math.max(0, (day.total - day.spam) / maxDaily * 78);
      const date = new Date(`${day.day}T00:00:00`);
      const label = Number.isNaN(date.getTime())
        ? day.day
        : date.toLocaleDateString([], { month: "numeric", day: "numeric" });
      return `<div class="trend-column" title="${escapeHTML(day.total)} scans"><b>${escapeHTML(day.total)}</b><div class="trend-stack"><i class="trend-spam" style="height:${spamHeight}px"></i><i class="trend-other" style="height:${otherHeight}px"></i></div><small>${escapeHTML(label)}</small></div>`;
    }).join("") : '<span class="empty-row">Your scan activity will appear here.</span>';
    const recent = $("#recent-scans");
    recent.innerHTML = stats.recent_scans.length ? stats.recent_scans.map(item => `
      <tr><td><b>${escapeHTML(item.subject || "(no subject)")}</b><small>${escapeHTML(item.sender || "Sender not provided")}</small></td>
      <td><span class="classification-pill ${escapeHTML(item.classification)}">${escapeHTML(classificationLabel(item.classification))}</span></td>
      <td><span class="risk-pill ${escapeHTML(item.risk_level)}">${escapeHTML(item.risk_level)}</span></td>
      <td>${escapeHTML(formatDate(item.created_at))}</td><td><a class="button button-quiet report-link" href="/api/history/${Number(item.id)}/report.pdf" aria-label="Download PDF report for scan ${Number(item.id)}">PDF report</a></td></tr>`).join("")
      : '<tr><td colspan="4" class="empty-row">No scans yet. Start by analyzing an email.</td></tr>';
  } catch (error) {
    toast(error.message);
  }

  try {
    const info = await requestJSON("/api/model-info");
    const metrics = info.metrics || {};
    $("#stat-spam-note").textContent = `Model decision threshold ${(Number(info.threshold) * 100).toFixed(0)}%`;
    $("#model-name").textContent = info.model_name || "Email model";
    $("#model-description").textContent = `${info.dataset_name || "Email corpus"} · ${Number(info.dataset_records_after_deduplication || 0).toLocaleString()} deduplicated records`;
    $("#metric-precision").textContent = percent(metrics.spam_precision);
    $("#metric-recall").textContent = percent(metrics.spam_recall);
    $("#metric-f1").textContent = percent(metrics.spam_f1);
  } catch (error) {
    $("#model-name").textContent = "Model unavailable";
    $("#model-description").textContent = error.message;
  }
}

function percent(value) {
  return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : "—";
}

function renderScanResult(result) {
  const container = $("#scan-result");
  $("#result-empty").hidden = true;
  container.hidden = false;
  const probabilityLabel = result.probability_type === "calibrated_sigmoid_probability"
    ? "Calibrated spam probability"
    : "Spam model score";
  const reasonItems = result.reasons.length
    ? result.reasons.map(reason => `<li>${escapeHTML(reason)}</li>`).join("")
    : "<li>No additional risk indicators were triggered.</li>";
  const phraseItems = result.suspicious_phrases.length
    ? result.suspicious_phrases.map(phrase => `<span class="phrase-chip">${escapeHTML(phrase)}</span>`).join("")
    : '<span class="muted-copy">None identified by the supplementary phrase rules.</span>';
  const urls = result.links.length
    ? result.links.map(link => `<li><div><b>${escapeHTML(link.domain)}</b><small>${escapeHTML(link.url)}</small></div>${link.flags.length ? `<span class="url-flags">${link.flags.map(escapeHTML).join(" · ")}</span>` : '<span class="url-safe">No URL pattern flag</span>'}</li>`).join("")
    : '<li class="muted-copy">No URLs detected.</li>';
  const attachments = result.attachment_analysis;
  const attachmentHtml = attachments
    ? '<div class="finding-section"><h3>Attachment inspection</h3>' +
      (attachments.items.length ? '<ul>' + attachments.items.map(item => '<li><b>' + escapeHTML(item.filename) + '</b> · ' + escapeHTML(item.content_type) + ' · ' + Number(item.size_bytes || 0).toLocaleString() + ' bytes · ' + escapeHTML(item.risk) + '</li>').join("") + '</ul>' : '<p class="muted-copy">No MIME attachments were found in this raw email.</p>') +
      (attachments.findings.length ? '<ul>' + attachments.findings.map(item => '<li>' + escapeHTML(item) + '</li>').join("") + '</ul>' : '') +
      '<small>Metadata-only inspection. Attachments are not opened, extracted, or executed.</small></div>'
    : "";
  const auth = result.email_authentication;
  const authenticationHtml = auth
    ? '<div class="finding-section"><h3>Email authentication</h3><p>SPF: <b>' + escapeHTML(auth.spf.toUpperCase()) + '</b> · DKIM: <b>' + escapeHTML(auth.dkim.toUpperCase()) + '</b> · DMARC: <b>' + escapeHTML(auth.dmarc.toUpperCase()) + '</b></p>' +
      (auth.findings.length ? '<ul>' + auth.findings.map(item => '<li>' + escapeHTML(item) + '</li>').join("") + '</ul>' : '<p class="muted-copy">No failure or reply-routing mismatch was found in the supplied headers.</p>') +
      '<small>' + escapeHTML(auth.note) + '</small></div>'
    : "";
  container.innerHTML = `
    <div class="result-topline"><span>ANALYSIS COMPLETE</span><span class="classification-pill ${escapeHTML(result.classification)}">${escapeHTML(result.classification_label)}</span></div>
    <div class="risk-heading"><div><span class="eyebrow">RISK LEVEL</span><h2 class="risk-text ${escapeHTML(result.risk_level)}">${escapeHTML(result.risk_level.toUpperCase())}</h2></div><div class="risk-score"><b>${escapeHTML(result.risk_score)}</b><small>Risk score / 100: Low 0-19, Medium 20-44, High 45-74, Critical 75-100</small></div></div>
    <div class="score-block"><div><span>${probabilityLabel}</span><b>${percent(result.spam_probability)}</b></div><div class="score-track"><i style="width:${Math.max(0, Math.min(100, result.spam_probability * 100))}%"></i></div><small>Decision threshold: ${percent(result.classification_threshold)} · review band: ${percent(result.uncertainty_band[0])}–${percent(result.uncertainty_band[1])}. This is not a probability of compromise.</small></div>
    <div class="finding-section"><h3>Why it was flagged</h3><ul>${reasonItems}</ul></div>
    <div class="finding-section"><h3>Suspicious phrases</h3><div class="phrase-list">${phraseItems}</div></div>
    <div class="finding-section"><h3>Links inspected <small>(not opened)</small></h3><ul class="url-list">${urls}</ul></div>
    ${authenticationHtml}
    ${attachmentHtml}
    <div class="recommendation"><b>Recommended action</b><p>${escapeHTML(result.recommended_action)}</p></div>`;
}

const scanForm = $("#scan-form");
scanForm?.addEventListener("submit", async event => {
  event.preventDefault();
  const button = $("#scan-submit");
  const rawEmail = $("#raw-email").value.trim();
  const body = $("#email-body").value.trim();
  const payload = rawEmail
    ? { raw_email: rawEmail }
    : {
        subject: $("#email-subject").value,
        body,
        sender: $("#email-sender").value,
        html_body: $("#email-html").value
      };
  if (!rawEmail && !payload.subject.trim() && !body && !payload.html_body.trim()) {
    toast("Enter email content or paste a raw email.");
    return;
  }
  button.disabled = true;
  button.textContent = "Analyzing…";
  try {
    const result = await requestJSON(rawEmail ? "/api/scan/raw" : "/api/scan", {
      method: "POST",
      body: JSON.stringify(payload)
    });
    renderScanResult(result);
    toast("Email analysis complete.");
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
    button.innerHTML = "Analyze email <span>→</span>";
  }
});

$("#email-body")?.addEventListener("input", event => {
  $("#body-count").textContent = event.currentTarget.value.length.toLocaleString();
});
$("#clear-scan")?.addEventListener("click", () => {
  $("#scan-form").reset();
  $("#body-count").textContent = "0";
  $("#scan-result").hidden = true;
  $("#result-empty").hidden = false;
});
document.querySelectorAll(".sample-button").forEach(button => button.addEventListener("click", () => {
  $("#raw-email").value = "";
  if (button.dataset.sample === "phishing") {
    $("#email-sender").value = "Account Support <alerts@account-security.example>";
    $("#email-subject").value = "Urgent: verify your account";
    $("#email-body").value = "Your account will be suspended today. Verify your password at http://192.0.2.1/login to avoid losing access.";
  } else {
    $("#email-sender").value = "Alex <alex@company.example>";
    $("#email-subject").value = "Updated schedule";
    $("#email-body").value = "The updated schedule is available in the shared folder. Let me know if you have questions.";
  }
  $("#body-count").textContent = $("#email-body").value.length.toLocaleString();
}));

const bulkForm = $("#bulk-form");
bulkForm?.addEventListener("submit", async event => {
  event.preventDefault();
  const file = $("#csv-file").files[0];
  if (!file) return toast("Choose a CSV file first.");
  const button = $("#bulk-submit");
  button.disabled = true;
  button.textContent = "Scanning…";
  try {
    const response = await fetch("/api/bulk", {
      method: "POST",
      body: new FormData(bulkForm),
      headers: { "X-CSRF-Token": csrfToken },
      credentials: "same-origin"
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Bulk scan failed.");
    $("#bulk-results").hidden = false;
    $("#bulk-summary").textContent = `${data.total} emails analyzed`;
    $("#bulk-export").href = `/api/bulk/${encodeURIComponent(data.batch_id)}/export`;
    $("#bulk-table").innerHTML = data.results.map(item => `
      <tr><td><b>${escapeHTML(item.subject || "(no subject)")}</b></td><td><span class="classification-pill ${escapeHTML(item.classification)}">${escapeHTML(item.classification_label)}</span></td><td><span class="risk-pill ${escapeHTML(item.risk_level)}">${escapeHTML(item.risk_level)}</span></td><td>${percent(item.spam_probability)}</td></tr>`).join("");
    const skipped = $("#bulk-skipped");
    skipped.hidden = !data.skipped_rows;
    skipped.textContent = data.skipped_rows ? `${data.skipped_rows} invalid or empty row(s) were skipped.` : "";
    toast("Bulk analysis complete.");
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
    button.innerHTML = "Scan CSV <span>→</span>";
  }
});

async function loadHistory() {
  const table = $("#history-table");
  if (!table) return;
  const params = new URLSearchParams();
  const query = $("#history-search").value.trim();
  const classification = $("#classification-filter").value;
  if (query) params.set("q", query);
  if (classification) params.set("classification", classification);
  try {
    const result = await requestJSON(`/api/history?${params}`);
    table.innerHTML = result.items.length ? result.items.map(item => `
      <tr><td><b>${escapeHTML(item.subject || "(no subject)")}</b><small>${escapeHTML(item.sender || "Sender not provided")}</small><small class="excerpt">${escapeHTML(item.text.slice(0, 130))}</small></td>
      <td><span class="classification-pill ${escapeHTML(item.classification)}">${escapeHTML(classificationLabel(item.classification))}</span></td>
      <td><span class="risk-pill ${escapeHTML(item.risk_level)}">${escapeHTML(item.risk_level)}</span></td><td>${percent(item.confidence)}</td><td>${escapeHTML(formatDate(item.created_at))}</td><td><a class="button button-quiet report-link" href="/api/history/${Number(item.id)}/report.pdf" aria-label="Download PDF report for scan ${Number(item.id)}">PDF report</a></td></tr>`).join("")
      : '<tr><td colspan="6" class="empty-row">No matching scans.</td></tr>';
    $("#history-count").textContent = `${result.items.length} scan${result.items.length === 1 ? "" : "s"} shown`;
  } catch (error) {
    table.innerHTML = `<tr><td colspan="6" class="empty-row">${escapeHTML(error.message)}</td></tr>`;
  }
}

$("#history-search")?.addEventListener("input", () => {
  window.clearTimeout(window.historyTimer);
  window.historyTimer = window.setTimeout(loadHistory, 200);
});
$("#classification-filter")?.addEventListener("change", loadHistory);
$("#clear-history")?.addEventListener("click", async () => {
  if (!window.confirm("Permanently delete all scan history for this account?")) return;
  try {
    const result = await requestJSON("/api/history", { method: "DELETE" });
    toast(`${result.deleted} records deleted.`);
    loadHistory();
  } catch (error) {
    toast(error.message);
  }
});

async function loadModelDetails() {
  if (!$("#about-model-name")) return;
  try {
    const info = await requestJSON("/api/model-info");
    const metrics = info.metrics || {};
    $("#about-model-name").textContent = info.model_name || "Email classifier";
    $("#about-model-data").textContent = `${info.dataset_name}; ${info.dataset_records_after_deduplication.toLocaleString()} email records; ${info.training_mode} training.`;
    $("#about-accuracy").textContent = percent(metrics.accuracy);
    $("#about-precision").textContent = percent(metrics.spam_precision);
    $("#about-recall").textContent = percent(metrics.spam_recall);
    $("#about-f1").textContent = percent(metrics.spam_f1);
    $("#about-roc").textContent = percent(metrics.roc_auc);
    $("#about-pr").textContent = percent(metrics.pr_auc);
  } catch (error) {
    $("#about-model-name").textContent = "Email model unavailable";
    $("#about-model-data").textContent = error.message;
  }
}


async function loadDomainRules() {
  const list = $("#domain-rule-list");
  if (!list) return;
  try {
    const result = await requestJSON("/api/domain-rules");
    list.innerHTML = result.items.length ? result.items.map(rule => `
      <div class="distribution-row">
        <span><b>${escapeHTML(rule.domain)}</b><small>${escapeHTML(formatDate(rule.created_at))}</small></span>
        <span class="classification-pill ${rule.action === "block" ? "phishing_suspicious" : "safe"}">${rule.action === "block" ? "BLOCK / FLAG" : "ALLOW / ANNOTATE"}</span>
        <button type="button" class="button button-quiet" data-delete-domain-rule="${Number(rule.id)}">Remove</button>
      </div>`).join("")
      : '<p class="empty-row">No custom domain rules yet.</p>';
  } catch (error) {
    list.innerHTML = `<p class="empty-row">${escapeHTML(error.message)}</p>`;
  }
}

$("#domain-rule-form")?.addEventListener("submit", async event => {
  event.preventDefault();
  const domain = $("#domain-rule-domain").value.trim();
  const action = $("#domain-rule-action").value;
  try {
    await requestJSON("/api/domain-rules", {
      method: "POST",
      body: JSON.stringify({ domain, action })
    });
    $("#domain-rule-domain").value = "";
    toast("Domain rule saved.");
    loadDomainRules();
  } catch (error) {
    toast(error.message);
  }
});

$("#domain-rule-list")?.addEventListener("click", async event => {
  const button = event.target.closest("[data-delete-domain-rule]");
  if (!button) return;
  try {
    await requestJSON(`/api/domain-rules/${button.dataset.deleteDomainRule}`, { method: "DELETE" });
    toast("Domain rule removed.");
    loadDomainRules();
  } catch (error) {
    toast(error.message);
  }
});

loadDashboard();
loadDomainRules();
loadHistory();
loadModelDetails();
