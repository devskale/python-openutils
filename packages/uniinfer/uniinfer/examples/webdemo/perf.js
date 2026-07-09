// UniInfer Performance Dashboard
// Measures TTFT, tokens/sec, and caching effects live via the proxy SSE stream.
// Shares history with `uniinfer --speedtest` through /perf/results.

const API_BASE = window.location.origin;
const STORAGE_KEY = "uniinfer_perf_api_key";
const DEFAULT_API_KEY = "";

// ─── Helpers ──────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const getApiKey = () => ($("api-key").value || "").trim();
const authHeader = () => ({ "Authorization": `Bearer ${getApiKey()}` });

const setStatus = (el, cls, text) => {
  el.className = `status-pill ${cls}`;
  el.textContent = text;
};

const fmt = (n, d = 2) =>
  typeof n === "number" && Number.isFinite(n) ? n.toFixed(d) : "–";

const nowISO = () => new Date().toISOString();

const randomSuffix = () => Math.random().toString(36).slice(2, 8);

// tok/s color coding (rough heuristics)
const tpsClass = (tps) =>
  tps >= 100 ? "good" : tps >= 30 ? "warn" : tps > 0 ? "bad" : "";

// TTFT color coding (seconds)
const ttftClass = (s) =>
  s <= 1 ? "good" : s <= 3 ? "warn" : "bad";

// ─── Model loading ─────────────────────────────────────────────
async function loadModels() {
  try {
    const res = await fetch(`${API_BASE}/v1/models`, { cache: "no-store", headers: authHeader() });
    const data = await res.json();
    const models = (data.data || [])
      .map((m) => m.id)
      .filter((id) => !/(tts|stt|whisper|image|embed|kokoro|piper)/i.test(id))
      .sort();
    const sel = $("model-select");
    sel.innerHTML = models
      .map((m) => `<option value="${m}">${m}</option>`)
      .join("");
    // Prefer a known-fast chat model if present
    const preferred = models.find((m) => /qwen-3\.6|gemma-4|qwen-3\.5/i.test(m));
    if (preferred) sel.value = preferred;
  } catch (e) {
    $("model-select").innerHTML =
      '<option value="">Modelle konnten nicht geladen werden</option>';
  }
}

// ─── Single streaming measurement ──────────────────────────────
// Returns { ttft, tft_thinking, tok_per_sec, text_tokens, thinking_tokens,
//           total_tokens, wall_time, finish_reason, error }
async function measureOnce({ model, prompt, maxTokens }) {
  if (!getApiKey()) return { error: "Kein API-Key gesetzt" };
  const body = {
    model,
    messages: [{ role: "user", content: prompt }],
    stream: true,
    max_tokens: maxTokens,
  };

  const start = performance.now();
  let firstTokenTime = null;
  let firstThinkingTime = null;
  let lastChunkTime = null;
  let textChars = 0;
  let thinkingChars = 0;
  let finishReason = null;
  let usage = null;

  try {
    const res = await fetch(`${API_BASE}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeader() },
      body: JSON.stringify(body),
    });
    if (!res.ok || !res.body) {
      const txt = await res.text().catch(() => "");
      return { error: `HTTP ${res.status}: ${txt.slice(0, 160)}` };
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop(); // keep incomplete line

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith(":")) continue; // heartbeat / comment
        if (!trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice(5).trim();
        if (payload === "[DONE]") continue;

        let chunk;
        try {
          chunk = JSON.parse(payload);
        } catch {
          continue;
        }
        if (chunk.error) {
          return { error: chunk.error.message || "stream error" };
        }
        if (chunk.usage) usage = chunk.usage;

        const choice = chunk.choices && chunk.choices[0];
        if (!choice) continue;
        const delta = choice.delta || {};

        if (delta.content) {
          if (firstTokenTime === null) firstTokenTime = performance.now();
          textChars += delta.content.length;
          lastChunkTime = performance.now();
        }
        if (delta.reasoning_content) {
          if (firstThinkingTime === null) firstThinkingTime = performance.now();
          thinkingChars += delta.reasoning_content.length;
          lastChunkTime = performance.now();
        }
        if (choice.finish_reason) {
          finishReason = choice.finish_reason;
          lastChunkTime = performance.now();
        }
      }
    }
  } catch (e) {
    return { error: e.message || String(e) };
  }

  const end = lastChunkTime || performance.now();

  // Token counting: prefer API usage, else ~4 chars/token estimate.
  let textTokens, thinkingTokens, totalTokens;
  if (usage && (usage.completion_tokens || usage.total_tokens)) {
    const completion = usage.completion_tokens || 0;
    const reasoning =
      (usage.completion_tokens_details &&
        usage.completion_tokens_details.reasoning_tokens) ||
      0;
    thinkingTokens = reasoning;
    textTokens = Math.max(0, completion - reasoning);
    totalTokens = usage.total_tokens || completion;
  } else {
    thinkingTokens = Math.round(thinkingChars / 4);
    textTokens = Math.round(textChars / 4);
    totalTokens = thinkingTokens + textTokens;
  }

  // TTFT = first of (content | thinking). TTFAT = first content token.
  const tftSource = firstTokenTime || firstThinkingTime;
  const ttft = tftSource ? (tftSource - start) / 1000 : 0;
  const tft_thinking = firstThinkingTime
    ? (firstThinkingTime - start) / 1000
    : null;

  const genTime = tftSource ? (end - tftSource) / 1000 : 0;
  const tokPerSec = genTime > 0 ? totalTokens / genTime : 0;

  return {
    ttft: Math.round(ttft * 1000) / 1000,
    tft_thinking:
      tft_thinking !== null ? Math.round(tft_thinking * 1000) / 1000 : null,
    tok_per_sec: Math.round(tokPerSec * 10) / 10,
    text_tokens: textTokens,
    thinking_tokens: thinkingTokens,
    total_tokens: totalTokens,
    wall_time: Math.round((end - start) / 10) / 100,
    finish_reason: finishReason,
  };
}

// ─── Render a single run row ───────────────────────────────────
function renderRunRow(label, r) {
  if (r.error) {
    const row = document.createElement("div");
    row.className = "run-row run-error";
    row.innerHTML = `<span class="run-label">${label}</span><span class="metric-value bad">⚠ ${r.error}</span>`;
    return row;
  }
  const row = document.createElement("div");
  row.className = "run-row";
  row.innerHTML = `
    <span class="run-label">${label}</span>
    <div class="metric"><span class="metric-label">TTFT</span><span class="metric-value ${ttftClass(r.ttft)}">${fmt(r.ttft)}s</span></div>
    ${r.tft_thinking !== null ? `<div class="metric"><span class="metric-label">TFT think</span><span class="metric-value">${fmt(r.tft_thinking)}s</span></div>` : ""}
    <div class="metric"><span class="metric-label">Tok/s</span><span class="metric-value ${tpsClass(r.tok_per_sec)}">${fmt(r.tok_per_sec, 1)}</span></div>
    <div class="metric"><span class="metric-label">Text tok</span><span class="metric-value">${r.text_tokens}</span></div>
    <div class="metric"><span class="metric-label">Think tok</span><span class="metric-value">${r.thinking_tokens}</span></div>
    <div class="metric"><span class="metric-label">Wall</span><span class="metric-value">${fmt(r.wall_time)}s</span></div>
    <div class="metric"><span class="metric-label">Finish</span><span class="metric-value" style="font-size:0.78rem">${r.finish_reason || "–"}</span></div>
  `;
  return row;
}

// ─── Live measurement (N runs) ────────────────────────────────
async function runLive() {
  const model = $("model-select").value;
  if (!model) return;
  const maxTokens = parseInt($("max-tokens").value, 10) || 1000;
  const runs = Math.min(parseInt($("run-count").value, 10) || 1, 10);
  const basePrompt = $("prompt-input").value || "Erkläre Attention.";
  const unique = $("unique-prompt").checked;

  const btn = $("run-btn");
  btn.disabled = true;
  const status = $("live-status");
  setStatus(status, "status-running", "läuft…");
  const out = $("live-results");
  out.innerHTML = "";

  const results = [];
  for (let i = 1; i <= runs; i++) {
    const prompt = unique ? `${basePrompt} [${randomSuffix()}]` : basePrompt;
    setStatus(status, "status-running", `Run ${i}/${runs}…`);
    const r = await measureOnce({ model, prompt, maxTokens });
    results.push(r);
    out.appendChild(renderRunRow(`Run ${i}`, r));
  }

  // Average + persist
  const ok = results.filter((r) => !r.error);
  if (ok.length > 0) {
    const avg = {};
    for (const k of ["ttft", "tok_per_sec", "thinking_tokens", "text_tokens", "total_tokens", "wall_time"]) {
      avg[k] = Math.round(
        (ok.reduce((s, r) => s + (r[k] || 0), 0) / ok.length) * 100
      ) / 100;
    }
    const thinkVals = ok.map((r) => r.tft_thinking).filter((v) => v !== null);
    if (thinkVals.length)
      avg.tft_thinking = Math.round((thinkVals.reduce((a, b) => a + b, 0) / thinkVals.length) * 1000) / 1000;
    avg.runs = ok.length;
    avg.finish_reason = ok[ok.length - 1].finish_reason;
    avg.tested_at = nowISO();

    const summary = document.createElement("div");
    summary.className = "run-row";
    summary.style.background = "#eff6ff";
    summary.style.borderColor = "#bfdbfe";
    summary.innerHTML = `
      <span class="run-label">Ø (${avg.runs})</span>
      <div class="metric"><span class="metric-label">TTFT</span><span class="metric-value ${ttftClass(avg.ttft)}">${fmt(avg.ttft)}s</span></div>
      ${avg.tft_thinking !== undefined ? `<div class="metric"><span class="metric-label">TFT think</span><span class="metric-value">${fmt(avg.tft_thinking)}s</span></div>` : ""}
      <div class="metric"><span class="metric-label">Tok/s</span><span class="metric-value ${tpsClass(avg.tok_per_sec)}">${fmt(avg.tok_per_sec, 1)}</span></div>
      <div class="metric"><span class="metric-label">Text tok</span><span class="metric-value">${avg.text_tokens}</span></div>
      <div class="metric"><span class="metric-label">Think tok</span><span class="metric-value">${avg.thinking_tokens}</span></div>
      <div class="metric"><span class="metric-label">Wall</span><span class="metric-value">${fmt(avg.wall_time)}s</span></div>
      <div class="metric"><span class="metric-label">Finish</span><span class="metric-value" style="font-size:0.78rem">${avg.finish_reason || "–"}</span></div>
    `;
    out.appendChild(summary);

    // Persist to shared history
    try {
      await fetch(`${API_BASE}/perf/results`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: model, result: avg }),
      });
      await loadHistory();
    } catch (e) {
      /* non-fatal */
    }
    setStatus(status, "status-ok", `${ok.length} Run(s) gespeichert`);
  } else {
    setStatus(status, "status-error", "alle Runs fehlerhaft");
  }
  btn.disabled = false;
}

// ─── Caching comparison ───────────────────────────────────────
async function runCacheTest() {
  const model = $("model-select").value;
  if (!model) return;
  const maxTokens = parseInt($("max-tokens").value, 10) || 1000;
  const prompt = $("cache-prompt").value;
  if (!prompt) return;

  const btn = $("cache-btn");
  btn.disabled = true;
  const status = $("cache-status");
  const out = $("cache-results");
  out.innerHTML = "";

  setStatus(status, "status-running", "Run 1/2 (cold)…");
  const cold = await measureOnce({ model, prompt, maxTokens });
  setStatus(status, "status-running", "Run 2/2 (warm)…");
  const warm = await measureOnce({ model, prompt, maxTokens });

  if (cold.error || warm.error) {
    out.innerHTML = `<div class="run-row run-error"><span class="run-label">Fehler</span><span class="metric-value bad">⚠ ${cold.error || warm.error}</span></div>`;
    setStatus(status, "status-error", "Fehler");
    btn.disabled = false;
    return;
  }

  const ttftDelta = cold.ttft - warm.ttft;
  const ttftPct = cold.ttft > 0 ? (ttftDelta / cold.ttft) * 100 : 0;
  const tpsDelta = warm.tok_per_sec - cold.tok_per_sec;

  const wrap = document.createElement("div");
  wrap.className = "cache-compare";
  wrap.innerHTML = `
    <div class="cache-cell cold">
      <div class="cache-cell-title">❄ Cold (Run 1)</div>
      <div class="metric"><span class="metric-label">TTFT</span><span class="metric-value ${ttftClass(cold.ttft)}">${fmt(cold.ttft)}s</span></div>
      <div class="metric" style="margin-top:8px"><span class="metric-label">Tok/s</span><span class="metric-value ${tpsClass(cold.tok_per_sec)}">${fmt(cold.tok_per_sec, 1)}</span></div>
    </div>
    <div class="cache-cell warm">
      <div class="cache-cell-title">🔥 Warm (Run 2)</div>
      <div class="metric"><span class="metric-label">TTFT</span><span class="metric-value ${ttftClass(warm.ttft)}">${fmt(warm.ttft)}s</span></div>
      <div class="metric" style="margin-top:8px"><span class="metric-label">Tok/s</span><span class="metric-value ${tpsClass(warm.tok_per_sec)}">${fmt(warm.tok_per_sec, 1)}</span></div>
    </div>
    <div class="cache-cell delta">
      <div class="cache-cell-title">Δ Caching-Effekt</div>
      <div class="metric"><span class="metric-label">TTFT-Ersparnis</span><span class="metric-value ${ttftDelta > 0.1 ? "good" : ttftDelta < -0.1 ? "bad" : ""}">${fmt(ttftDelta)}s (${fmt(ttftPct, 0)}%)</span></div>
      <div class="metric" style="margin-top:8px"><span class="metric-label">Tok/s-Änderung</span><span class="metric-value">${tpsDelta >= 0 ? "+" : ""}${fmt(tpsDelta, 1)}</span></div>
    </div>
  `;
  out.appendChild(wrap);

  const note = document.createElement("p");
  note.className = "card-hint";
  note.style.marginTop = "12px";
  if (ttftDelta > 0.2) {
    note.innerHTML = `✅ <strong>Caching wirkt</strong>: TTFT ist um ${fmt(ttftPct, 0)}% gesunken. Der Provider hat den Prompt-Prefix gecacht (Prefill geskipped). Output-Speed bleibt erwartungsgemäß ${tpsDelta < 5 ? "konstant" : "leicht verändert"}.`;
  } else if (ttftDelta < -0.2) {
    note.innerHTML = `⚠ <strong>Kein Caching</strong> (oder Messrauschen): Warm war langsamer als Cold. Provider cacht diesen Prompt vermutlich nicht, oder Last schwankt.`;
  } else {
    note.innerHTML = `➖ <strong>Kein nennenswerter Caching-Effekt</strong>: TTFT-Differenz < 0.2s. Entweder cacht der Provider nicht, oder der Prompt-Prefix ist zu kurz.`;
  }
  out.appendChild(note);

  setStatus(status, "status-ok", "fertig");
  btn.disabled = false;
}

// ─── History table ─────────────────────────────────────────────
async function loadHistory() {
  try {
    const res = await fetch(`${API_BASE}/perf/results`, { cache: "no-store" });
    const data = await res.json();
    const entries = Object.entries(data).sort((a, b) =>
      String(a[0]).localeCompare(String(b[0]))
    );
    $("history-count").textContent = `${entries.length} Modell(e)`;

    if (entries.length === 0) {
      $("history-table").innerHTML =
        '<table class="perf-table"><tr><td class="empty">Noch keine Messungen. Starte einen Live-Test oder <code>uniinfer --speedtest</code>.</td></tr></table>';
      return;
    }

    // Find max tok/s for bar scaling
    const maxTps = Math.max(
      ...entries.map(([, v]) => v.tok_per_sec || 0),
      1
    );

    const rows = entries
      .map(([key, v]) => {
        const tpsPct = Math.round(((v.tok_per_sec || 0) / maxTps) * 100);
        const tested = v.tested_at
          ? new Date(v.tested_at).toLocaleString("de-AT", {
              day: "2-digit",
              month: "2-digit",
              hour: "2-digit",
              minute: "2-digit",
            })
          : "–";
        return `<tr>
          <td class="model-cell">${key}</td>
          <td class="${ttftClass(v.tft || 0)}">${fmt(v.tft)}s</td>
          ${v.tft_thinking !== null && v.tft_thinking !== undefined ? `<td>${fmt(v.tft_thinking)}s</td>` : "<td>–</td>"}
          <td class="num ${tpsClass(v.tok_per_sec || 0)}">
            <div class="bar-wrap">
              <span>${fmt(v.tok_per_sec, 1)}</span>
              <span class="bar-track"><span class="bar-fill" style="width:${tpsPct}%"></span></span>
            </div>
          </td>
          <td class="num">${v.text_tokens ?? "–"}</td>
          <td class="num">${v.thinking_tokens ?? "–"}</td>
          <td class="num">${v.total_tokens ?? "–"}</td>
          <td class="num">${fmt(v.wall_time)}s</td>
          <td>${v.runs ?? "–"}</td>
          <td style="font-size:0.72rem;color:var(--text-muted)">${tested}</td>
        </tr>`;
      })
      .join("");

    $("history-table").innerHTML = `
      <table class="perf-table">
        <thead><tr>
          <th>Modell</th>
          <th>TTFT</th>
          <th>TFT think</th>
          <th>Tok/s</th>
          <th class="num">Text</th>
          <th class="num">Think</th>
          <th class="num">Total</th>
          <th class="num">Wall</th>
          <th class="num">Runs</th>
          <th>Getestet</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  } catch (e) {
    $("history-table").innerHTML =
      '<table class="perf-table"><tr><td class="empty">Historie konnte nicht geladen werden.</td></tr></table>';
  }
}

// ─── Wire up ───────────────────────────────────────────────────
$("run-btn").addEventListener("click", runLive);
$("cache-btn").addEventListener("click", runCacheTest);
$("refresh-history").addEventListener("click", loadHistory);

// API key: load from localStorage (or default), persist on change.
(function initApiKey() {
  const stored = localStorage.getItem(STORAGE_KEY);
  $("api-key").value = stored || DEFAULT_API_KEY;
  $("api-key").addEventListener("change", () => {
    if ($("remember-key").checked) {
      localStorage.setItem(STORAGE_KEY, getApiKey());
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  });
  $("remember-key").addEventListener("change", () => {
    if ($("remember-key").checked) {
      localStorage.setItem(STORAGE_KEY, getApiKey());
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  });
})();

loadModels();
loadHistory();
