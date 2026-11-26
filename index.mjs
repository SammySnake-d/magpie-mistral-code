// Mistral Vibe Code, as an OpenCode provider plugin (unofficial), served the
// way the official mistral-vibe CLI serves it:
//
// - Sign-in is the browser sign-in the CLI does: POST console.mistral.ai's
//   /api/vibe/sign-in with a PKCE challenge, the page it answers with is
//   opened, its poll_url is read until the browser has signed in, and
//   POST /api/vibe/sign-in/{process_id}/exchange (exchange_token +
//   code_verifier) returns an API key. That key is made for Vibe Code: its
//   calls on api.mistral.ai bill the plan's monthly Vibe Code budget, not
//   pay-as-you-go API credits. A key made the same way in the console
//   (Code > Vibe Code CLI) can be pasted instead.
// - Requests are plain OpenAI chat completions on https://api.mistral.ai/v1
//   with the key as a bearer token; the models are /v1/models. No private
//   protocol (the CLI's ACP layer is its IDE control plane, not its model
//   channel), so the loader re-signs each request, and puts an earlier
//   turn's thinking back in the typed part Mistral takes it (see
//   mistralBody below): Mistral turns an unknown message field away as 422
//   extra_forbidden.
// - Usage is the public billing read the CLI does:
//   GET api.mistral.ai/v1/billing/subscription (plan name, pay-as-you-go) and
//   GET api.mistral.ai/v1/billing/budget (vibe_budget: usage_percentage,
//   initial_budget, reset_at); the plan also comes from console's
//   /api/vibe/whoami.
//
// The provider id is "mistral-vibe", kept apart from the "mistral" preset
// (api.mistral.ai with the user's own API key).
import { createHash, randomBytes } from "node:crypto"

const ID = "mistral-vibe"

// bases are functions so tests can point them at a stand-in with the
// MISTRAL_VIBE_API_BASE / MISTRAL_VIBE_CONSOLE_API environment variables
const apiBase = () => (process.env.MISTRAL_VIBE_API_BASE || "https://api.mistral.ai/v1").replace(/\/+$/, "")
const consoleApi = () => (process.env.MISTRAL_VIBE_CONSOLE_API || "https://console.mistral.ai/api").replace(/\/+$/, "")

const POLL_EVERY_MS = 3000
const RATE_LIMIT_MAX_WAIT_MS = 30_000
const POLL_429_MAX_ATTEMPTS = 6
const EXCHANGE_MAX_ATTEMPTS = 6
const SIGN_IN_TTL_MS = 10 * 60 * 1000
const DEFAULT_LIMIT = { context: 262_144, output: 65_536 }
const DEFAULT_UA = "mistral-client-python/Mistral-Vibe/3.1.3"

const userAgent = () => process.env.MISTRAL_VIBE_UA || DEFAULT_UA
const pollEveryMs = () => {
  const n = Number(process.env.MISTRAL_VIBE_POLL_MS)
  return Number.isFinite(n) && n > 0 ? n : POLL_EVERY_MS
}
const jsonHeaders = (extra) => ({ accept: "application/json", "user-agent": userAgent(), ...extra })
const signInFetch = (url, init = {}) =>
  fetch(url, {
    ...init,
    headers: jsonHeaders({ connection: "close", ...init.headers }),
    keepalive: false,
  })
const loggerOf = (client) => (level, message) => {
  try {
    return Promise.resolve(client?.app?.log?.({ body: { service: ID, level, message } })).catch(() => {})
  } catch {}
}

// Prices from https://docs.mistral.ai/inference/pricing (USD per 1M tokens,
// standard tier, checked 2026-11-06), for the current LLM models only. The
// keys are the ids the docs and /v1/models agree on, plus Large 4's billing
// name; an alias is priced through its own billing_model_name, so it follows
// the model behind it rather than a price kept for the alias. Older
// generations and the speech, OCR and embedding models keep models.dev's
// price when it has one.
const DOCS_PRICES = {
  "mistral-large-4": { input: 0.7, output: 2.1, cache_read: 0.07 },
  "mistral-large-4-0": { input: 0.7, output: 2.1, cache_read: 0.07 },
  "mistral-large-4-0-a": { input: 0.7, output: 2.1, cache_read: 0.07 },
  "mistral-small-2603": { input: 0.15, output: 0.6, cache_read: 0.015 },
  "ministral-14b-2512": { input: 0.2, output: 0.2, cache_read: 0.02 },
  "ministral-8b-2512": { input: 0.15, output: 0.15, cache_read: 0.015 },
  "ministral-3b-2512": { input: 0.1, output: 0.1, cache_read: 0.01 },
  "codestral-2508": { input: 0.3, output: 0.9, cache_read: 0.03 },
  "zai-glm-5-3": { input: 1.4, output: 4.4, cache_read: 0.14 },
  "zai-glm-5-3-flash": { input: 0.15, output: 0.5, cache_read: 0.03 },
  "deepseek-v4-1-flash-1017": { input: 0.3, output: 1.2, cache_read: 0.007 },
  // mistral-large-latest is deliberately absent: the vendor still bills it as
  // mistral-large-2512 (Large 3), so models.dev's price for it is the right one.
}

const sleep = (ms, signal) =>
  new Promise((resolve, reject) => {
    const t = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort)
      resolve()
    }, ms)
    const onAbort = () => {
      clearTimeout(t)
      reject(signal.reason ?? new DOMException("Aborted", "AbortError"))
    }
    if (signal?.aborted) onAbort()
    else signal?.addEventListener("abort", onAbort, { once: true })
  })

const keyOf = (auth) => (auth?.type === "api" || auth?.key ? auth.key : auth?.access) || ""

// ---- the browser sign-in (vibe's setup/auth/http_browser_sign_in_gateway) ----

const newVerifier = () => randomBytes(48).toString("base64url")
const challengeOf = (verifier) => createHash("sha256").update(verifier).digest("base64url")

const WORKSPACE_NOT_ELIGIBLE_CODE = 9004

function rateLimitWaitMs(res, previousMs = 0, baseMs = pollEveryMs()) {
  const raw = res?.headers?.get?.("retry-after")?.trim()
  if (raw) {
    const sec = Number(raw)
    if (Number.isFinite(sec)) {
      if (sec > 0) return Math.min(Math.ceil(sec * 1000), RATE_LIMIT_MAX_WAIT_MS)
    } else {
      const at = Date.parse(raw)
      if (Number.isFinite(at)) {
        const delta = at - Date.now()
        if (delta > 0) return Math.min(delta, RATE_LIMIT_MAX_WAIT_MS)
      }
    }
  }
  return Math.min(Math.max(previousMs * 2, baseMs * 2), RATE_LIMIT_MAX_WAIT_MS)
}

async function readErrorInfo(res) {
  try {
    const text = await res.text()
    if (!text) return { errorCode: undefined, detail: "" }
    try {
      const j = JSON.parse(text)
      let detail = ""
      for (const k of ["detail", "message", "error"]) {
        if (typeof j?.[k] === "string" && j[k].trim()) {
          detail = j[k].trim().slice(0, 200)
          break
        }
      }
      return { errorCode: j?.error_code, detail }
    } catch {}
    return { errorCode: undefined, detail: text.trim().slice(0, 200) }
  } catch {
    return { errorCode: undefined, detail: "" }
  }
}

async function startSignIn(signal) {
  const code_verifier = newVerifier()
  const res = await signInFetch(`${consoleApi()}/vibe/sign-in`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ code_challenge: challengeOf(code_verifier), code_challenge_method: "S256" }),
    signal,
  })
  if (!res.ok) {
    const { detail } = await readErrorInfo(res)
    const suffix = res.status === 429 ? " (rate-limited; switch proxy node or wait before trying again)" : ""
    throw new Error(`Mistral sign-in could not start (${res.status}${detail ? `: ${detail}` : ""})${suffix}`)
  }
  const d = await res.json()
  if (!d?.process_id || !d?.sign_in_url || !d?.poll_url) throw new Error("Mistral sign-in answered an unexpected payload")
  return { process_id: d.process_id, sign_in_url: d.sign_in_url, poll_url: d.poll_url, expires_at: d.expires_at, code_verifier }
}

async function pollSignIn(pollUrl, signal) {
  const res = await signInFetch(pollUrl, { signal })
  if (res.status === 410) {
    await res.body?.cancel?.().catch(() => {})
    return { status: "expired" }
  }
  if (res.status === 429) {
    const { detail } = await readErrorInfo(res)
    const cfRay = res.headers?.get?.("cf-ray") || ""
    return { status: "rate_limited", response: res, detail, cfRay }
  }
  if (!res.ok) {
    const { detail } = await readErrorInfo(res)
    const err = new Error(`Mistral sign-in poll answered ${res.status}${detail ? `: ${detail}` : ""}`)
    err.status = res.status
    throw err
  }
  return await res.json()
}

async function exchangeSignIn(processId, exchangeToken, codeVerifier, signal, deadline = Date.now() + 2 * 60 * 1000, log) {
  const url = `${consoleApi()}/vibe/sign-in/${encodeURIComponent(processId)}/exchange`
  const payload = JSON.stringify({ exchange_token: exchangeToken, code_verifier: codeVerifier })
  let wait = 0
  let lastStatus = 0
  let lastDetail = ""

  for (let attempt = 1; attempt <= EXCHANGE_MAX_ATTEMPTS; attempt++) {
    let res
    try {
      res = await signInFetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: payload,
        signal,
      })
    } catch (e) {
      if (signal?.aborted || attempt >= EXCHANGE_MAX_ATTEMPTS) throw e
      wait = Math.min(Math.max(wait * 2, pollEveryMs()), RATE_LIMIT_MAX_WAIT_MS)
      const remaining = deadline - Date.now()
      if (remaining <= 0) throw e
      const sleepMs = Math.min(wait, remaining)
      log?.("warn", `sign-in exchange network error (${attempt}/${EXCHANGE_MAX_ATTEMPTS}) wait=${sleepMs}ms: ${e?.message ?? e}`)
      await sleep(sleepMs, signal)
      continue
    }

    if (res.ok) {
      const d = await res.json()
      if (!d?.api_key) throw new Error("Mistral sign-in exchange returned no API key")
      return d.api_key
    }

    const { errorCode, detail } = await readErrorInfo(res)
    lastStatus = res.status
    lastDetail = detail
    if (res.status === 403 && errorCode === WORKSPACE_NOT_ELIGIBLE_CODE) {
      throw new Error("Current Mistral workspace is not eligible for Mistral Vibe; sign in again and select another workspace")
    }

    const retryable = res.status === 429 || res.status === 502 || res.status === 503 || res.status === 504
    if (!retryable || attempt >= EXCHANGE_MAX_ATTEMPTS) {
      throw new Error(`Mistral sign-in exchange failed (${res.status}${detail ? `: ${detail}` : ""})`)
    }

    wait = res.status === 429 ? rateLimitWaitMs(res, wait, pollEveryMs()) : Math.min(Math.max(wait * 2, pollEveryMs()), RATE_LIMIT_MAX_WAIT_MS)
    const remaining = deadline - Date.now()
    if (remaining <= 0) {
      throw new Error(`Mistral sign-in exchange failed (${lastStatus}${lastDetail ? `: ${lastDetail}` : ""})`)
    }
    const sleepMs = Math.min(wait, remaining)
    const cfRay = res.headers?.get?.("cf-ray") || ""
    log?.("warn", `sign-in exchange ${res.status} (${attempt}/${EXCHANGE_MAX_ATTEMPTS}) wait=${sleepMs}ms${cfRay ? ` cf-ray=${cfRay}` : ""}${detail ? ` detail=${detail}` : ""}`)
    await sleep(sleepMs, signal)
  }

  throw new Error(`Mistral sign-in exchange failed (${lastStatus || "unknown"}${lastDetail ? `: ${lastDetail}` : ""})`)
}

// whoami names the plan behind a key (console.mistral.ai/api/vibe/whoami)
async function whoami(key) {
  const res = await signInFetch(`${consoleApi()}/vibe/whoami`, {
    headers: { Authorization: `Bearer ${key}` },
    signal: AbortSignal.timeout(8000),
  })
  if (!res.ok) return null
  return await res.json().catch(() => null)
}

// ---- models: /v1/models, limits filled from models.dev when it knows them ----

let devModels = null
async function modelsDev() {
  if (devModels) return devModels
  try {
    const res = await fetch("https://models.dev/api.json", { signal: AbortSignal.timeout(5000) })
    const j = await res.json()
    devModels = j?.mistral?.models ?? {}
  } catch {
    devModels = {}
  }
  return devModels
}

// priceOf is a model's list price as OpenCode and magpie take it: USD per
// 1M tokens, with the cache read and write beside input and output. docs
// first, then models.dev, and undefined when nothing prices it rather than
// a made-up zero.
function priceOf(m, dev) {
  const caps = m?.capabilities ?? {}
  const billing = typeof m?.billing_model_name === "string" ? m.billing_model_name : ""
  const aliases = Array.isArray(m?.aliases) ? m.aliases : []
  const names = [billing, ...aliases, m?.id].filter((id) => typeof id === "string" && id)
  const stated = names.map((id) => DOCS_PRICES[id]).find(Boolean)
  // a speech or non-chat model's models.dev price may be per minute, page or
  // character rather than per token, so only chat models without audio take it
  const audio =
    caps.audio === true || caps.audio_transcription === true || caps.audio_transcription_realtime === true || caps.audio_speech === true
  const chat = caps.completion_chat ?? true
  const p = stated ?? (chat && !audio ? names.map((id) => dev?.[id]?.cost).find(Boolean) : undefined)
  if (!p || typeof p !== "object") return undefined
  const n = (v) => (Number.isFinite(Number(v)) && Number(v) >= 0 ? Number(v) : 0)
  const cache = p.cache && typeof p.cache === "object" ? p.cache : {}
  return {
    input: n(p.input),
    output: n(p.output),
    cache: { read: n(p.cache_read ?? cache.read), write: n(p.cache_write ?? cache.write) },
  }
}

async function listModels(key) {
  const res = await fetch(`${apiBase()}/models`, {
    headers: jsonHeaders({ Authorization: `Bearer ${key}` }),
    signal: AbortSignal.timeout(15_000),
  })
  if (!res.ok) return null
  const j = await res.json().catch(() => null)
  const dev = await modelsDev()
  const out = {}
  for (const m of j?.data ?? []) {
    if (!m?.id || typeof m.id !== "string") continue
    const billing = typeof m.billing_model_name === "string" ? m.billing_model_name : ""
    const aliases = Array.isArray(m.aliases) ? m.aliases : []
    const d = [billing, ...aliases, m.id].map((id) => dev[id]).find(Boolean) ?? {}
    const caps = m.capabilities ?? {}
    const image = caps.vision ?? d.attachment ?? false
    const reasoning = caps.reasoning ?? d.reasoning ?? false
    const toolcall = caps.function_calling ?? d.tool_call ?? true
    const chat = caps.completion_chat ?? true
    const context =
      Number(m.max_context_length) > 0 ? Number(m.max_context_length) : d.limit?.context ?? DEFAULT_LIMIT.context
    const inputs = {
      text: caps.ocr !== true,
      image: image === true,
      audio: caps.audio === true || caps.audio_transcription === true || caps.audio_transcription_realtime === true,
      video: false,
      pdf: Array.isArray(d.modalities?.input) && d.modalities.input.includes("pdf"),
    }
    const outputs = { text: chat === true, image: false, audio: caps.audio_speech === true, video: false, pdf: false }
    const modalities = (kinds) => Object.keys(kinds).filter((k) => kinds[k])
    const model = {
      name: d.name ?? m.id,
      limit: {
        context,
        output: d.limit?.output ?? DEFAULT_LIMIT.output,
      },
      tool_call: toolcall,
      attachment: image,
      reasoning,
      temperature: chat,
      // OpenCode 1.x's model shape, so magpie and 2.x see what the model
      // takes and gives (and leave a TTS or an embedding out of chat lists)
      modalities: { input: modalities(inputs), output: modalities(outputs) },
      capabilities: {
        temperature: chat,
        reasoning,
        attachment: image,
        toolcall,
        input: inputs,
        output: outputs,
        interleaved: false,
      },
    }
    const cost = priceOf(m, dev)
    if (cost) model.cost = cost
    out[m.id] = model
  }
  return out
}

// ---- usage: the billing reads (vibe's app_server/_billing.py) ----

async function readUsage(key) {
  const headers = jsonHeaders({ Authorization: `Bearer ${key}` })
  const ask = async (url) => {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(8000) })
    if (!res.ok) return null
    return await res.json().catch(() => null)
  }
  const [subscription, budget, who] = await Promise.all([
    ask(`${apiBase()}/billing/subscription`),
    ask(`${apiBase()}/billing/budget`),
    whoami(key),
  ])
  const plan = subscription?.plan_name || who?.plan_name || undefined
  const vb = budget?.vibe_budget
  const ab = budget?.api_budget
  const usd = (x) => Math.round(x * 1e6) / 1e6
  const windows = []
  const windowOf = (name, w, aside) => {
    if (!w || typeof w.usage_percentage !== "number") return null
    const initial = typeof w.initial_budget === "number" ? w.initial_budget : 0
    const resetsAt = w.reset_at ? Date.parse(w.reset_at) : NaN
    const span = Number.isFinite(resetsAt) ? Math.max(3600, Math.floor((resetsAt - Date.now()) / 1000)) : 30 * 86400
    const usedUsd = usd(initial * w.usage_percentage / 100)
    const out = {
      name,
      used: w.usage_percentage,
      amount: usedUsd,
      limit: initial,
      unit: "USD",
      resetsAt: Number.isFinite(resetsAt) ? resetsAt : undefined,
      span,
    }
    if (aside) out.aside = true
    return out
  }
  const vbm = windowOf("Monthly Vibe Code", vb, false)
  if (vbm) windows.push(vbm)
  const abm = windowOf("Platform API credits", ab, true)
  if (abm) windows.push(abm)
  return { plan, windows, payg: subscription?.vibe_payg_enabled ?? vb?.payg_enabled ?? undefined }
}

// ---- an earlier turn's thinking, in the shape Mistral takes it ----

// api.mistral.ai reads each chat message strictly: a field it doesn't know
// turns the whole request away as 422 "Extra inputs are not permitted"
// (extra_forbidden, loc ["body","messages",2,"assistant",
// "reasoning_content"]). An earlier assistant turn's thinking comes back
// from most clients in one of these fields — reasoning_content (OpenCode,
// pi, Claude Code through a gateway), reasoning (OpenRouter's) and
// reasoning_details (OpenRouter's structured one) — so a conversation with
// a thinking turn in it never gets through. Mistral returns a thinking turn
// as a typed content part and takes it back that way, so the thinking goes
// there, ahead of the turn's own text, as magpie's own Mistral handling
// gives it (#494).
const THINKING_FIELDS = ["reasoning_content", "reasoning", "reasoning_details"]

// thinkingText is the thinking a message carries in THINKING_FIELDS.
function thinkingText(m) {
  for (const f of THINKING_FIELDS.slice(0, 2)) {
    const v = m[f]
    if (typeof v === "string" && v !== "") return v
  }
  const details = m.reasoning_details
  if (!Array.isArray(details)) return ""
  let out = ""
  for (const d of details) {
    if (!d || typeof d !== "object") continue
    if (typeof d.text === "string" && d.text !== "") out += d.text
    else if (typeof d.summary === "string") out += d.summary
  }
  return out
}

// MISTRAL_MESSAGE_FIELDS are the fields Mistral takes on a chat message; the
// rest it turns away, the same extra_forbidden way (#1: WorkBuddy 5.11.3 puts
// startsNewUserRequest / agent / conversationRequestId on every message it
// sends, so loc ["body","messages",1,"user","startsNewUserRequest"]).
const MISTRAL_MESSAGE_FIELDS = new Set(["role", "content", "name", "prefix", "tool_calls", "tool_call_id"])

// asMistralMessage puts one message's thinking where Mistral takes it, drops
// the fields Mistral doesn't take, and says whether it changed anything. A
// message with nothing to move keeps its bytes. A prefill (prefix) keeps its
// content as it is: its text is what goes on, so the thinking is only
// dropped there.
function asMistralMessage(m) {
  let changed = false
  for (const f of Object.keys(m)) {
    if (!MISTRAL_MESSAGE_FIELDS.has(f) && !THINKING_FIELDS.includes(f)) {
      delete m[f]
      changed = true
    }
  }
  if (!THINKING_FIELDS.some((f) => f in m)) return changed
  const think = thinkingText(m)
  for (const f of THINKING_FIELDS) delete m[f]
  if (think === "" || m.role !== "assistant" || m.prefix === true) return true
  const part = { type: "thinking", thinking: [{ type: "text", text: think }] }
  const c = m.content
  if (typeof c === "string") m.content = c === "" ? [part] : [part, { type: "text", text: c }]
  else if (Array.isArray(c)) m.content = [part, ...c]
  else m.content = [part]
  return true
}

// mistralBody is a chat request's body with every message's thinking in the
// shape Mistral takes, and without the fields it turns away. A body with
// nothing to move comes back as it was, byte for byte, and so does one that
// isn't a chat request's JSON.
function mistralBody(body) {
  if (typeof body !== "string") return body
  let q
  try {
    q = JSON.parse(body)
  } catch {
    return body
  }
  if (!q || typeof q !== "object" || !Array.isArray(q.messages)) return body
  let changed = false
  for (const m of q.messages) if (m && typeof m === "object" && asMistralMessage(m)) changed = true
  return changed ? JSON.stringify(q) : body
}

// ---- the plugin ----

export const MistralVibeAuthPlugin = async ({ client } = {}) => {
  const log = loggerOf(client)
  return {
  config: async (cfg) => {
    cfg.provider ??= {}
    cfg.provider[ID] ??= {
      name: "Mistral Vibe Code",
      npm: "@ai-sdk/openai-compatible",
      api: apiBase(),
      models: {},
    }
  },
  auth: {
    provider: ID,
    usage: async (getAuth) => {
      const key = keyOf(await getAuth())
      if (!key) return { error: "Not signed in", windows: [] }
      try {
        const u = await readUsage(key)
        return { plan: u.plan, windows: u.windows }
      } catch (e) {
        return { error: `Mistral billing read failed: ${e?.message ?? e}`, windows: [] }
      }
    },
    loader: async (getAuth) => {
      const auth = await getAuth()
      const key = keyOf(auth)
      if (!key) return {}
      return {
        baseURL: apiBase(),
        apiKey: "mistral-vibe", // the engine's placeholder; the fetch below signs
        async fetch(input, init = {}) {
          const headers = new Headers(init.headers ?? (input instanceof Request ? input.headers : undefined))
          headers.set("Authorization", `Bearer ${key}`)
          // 与官方 CLI 一致的客户端标识（mistral-vibe vibe/utils/http.py: get_user_agent）
          if (!headers.has("user-agent")) headers.set("user-agent", userAgent())
          const body = mistralBody(init.body)
          if (body !== init.body) {
            // the length the engine counted is not the one sent
            headers.delete("content-length")
            return fetch(input, { ...init, headers, body })
          }
          return fetch(input, { ...init, headers })
        },
      }
    },
    methods: [
      {
        type: "oauth",
        label: "Sign in with Mistral (browser)",
        authorize: async (_inputs, { signal } = {}) => {
          let p
          try {
            p = await startSignIn(signal)
          } catch (e) {
            log("error", e?.message ?? String(e))
            throw e
          }
          log("info", `sign-in started process=${p.process_id}`)
          return {
            url: p.sign_in_url,
            instructions: "Sign in to your Mistral account on the page that opens. The sign-in finishes here by itself.",
            method: "auto",
            callback: async (_code, { signal: cbSignal } = {}) => {
              const s = cbSignal ?? (signal?.aborted ? undefined : signal)
              const deadline = Date.parse(p.expires_at) || Date.now() + SIGN_IN_TTL_MS
              let rateWait = 0
              let rate429Count = 0
              let pollFailures = 0
              let loggedPending = false

              for (;;) {
                if (Date.now() > deadline) {
                  log("warn", `sign-in timed out process=${p.process_id}`)
                  return { type: "failed", error: "The sign-in timed out; start it again" }
                }
                let r
                try {
                  r = await pollSignIn(p.poll_url, s)
                } catch (e) {
                  if (s?.aborted) throw e
                  pollFailures++
                  log("warn", `sign-in poll error (${pollFailures}) status=${e?.status ?? "net"}: ${e?.message ?? e}`)
                  if (Date.now() > deadline) return { type: "failed", error: `Sign-in poll failed: ${e?.message ?? e}` }
                  if (e?.status >= 400 && e?.status < 500 && e?.status !== 408 && pollFailures >= 3) {
                    return { type: "failed", error: `Sign-in poll failed: ${e?.message ?? e}` }
                  }
                  const wait = Math.min(pollEveryMs() * Math.min(pollFailures, 4), RATE_LIMIT_MAX_WAIT_MS)
                  const remaining = deadline - Date.now()
                  if (remaining <= 0) return { type: "failed", error: `Sign-in poll failed: ${e?.message ?? e}` }
                  await sleep(Math.min(wait, remaining), s)
                  continue
                }

                if (r?.status === "rate_limited") {
                  rate429Count++
                  rateWait = rateLimitWaitMs(r.response, rateWait, pollEveryMs())
                  const remaining = deadline - Date.now()
                  const sleepMs = Math.min(rateWait, Math.max(0, remaining))
                  log("warn", `sign-in poll 429 (${rate429Count}/${POLL_429_MAX_ATTEMPTS}) wait=${sleepMs}ms${r.cfRay ? ` cf-ray=${r.cfRay}` : ""}${r.detail ? ` detail=${r.detail}` : ""}`)
                  if (rate429Count >= POLL_429_MAX_ATTEMPTS) {
                    return {
                      type: "failed",
                      error: `Sign-in poll rate-limited (429${r.detail ? `: ${r.detail}` : ""}): switch proxy node or wait before retrying`,
                    }
                  }
                  if (remaining <= 0) return { type: "failed", error: "The sign-in timed out; start it again" }
                  await sleep(sleepMs, s)
                  continue
                }

                rateWait = 0
                rate429Count = 0
                pollFailures = 0

                if (r?.status === "completed") {
                  if (!r.exchange_token) return { type: "failed", error: "Mistral sign-in completed without exchange token" }
                  log("info", `sign-in poll completed process=${p.process_id}; exchanging token`)
                  try {
                    const key = await exchangeSignIn(p.process_id, r.exchange_token, p.code_verifier, s, deadline, log)
                    const who = await whoami(key).catch(() => null)
                    log("info", `sign-in succeeded process=${p.process_id} plan=${who?.plan_name ?? "unknown"}`)
                    const out = { type: "success", key, metadata: { plan: who?.plan_name, vibeEntitlement: who?.has_paid_vibe_entitlement } }
                    if (who?.customer_id != null) out.accountId = String(who.customer_id)
                    return out
                  } catch (e) {
                    if (s?.aborted) throw e
                    log("error", `sign-in exchange failed process=${p.process_id}: ${e?.message ?? e}`)
                    return { type: "failed", error: `Sign-in exchange failed: ${e?.message ?? e}` }
                  }
                }
                if (r?.status === "expired") {
                  log("warn", `sign-in expired process=${p.process_id}`)
                  return { type: "failed", error: "The sign-in expired; start it again" }
                }
                if (r?.status === "denied") {
                  log("warn", `sign-in denied process=${p.process_id}`)
                  return { type: "failed", error: r.message || "The sign-in was denied" }
                }
                if (r?.status === "error") {
                  log("warn", `sign-in error process=${p.process_id}: ${r.message ?? "unknown"}`)
                  return { type: "failed", error: r.message || "The sign-in failed" }
                }
                if (!loggedPending) {
                  loggedPending = true
                  log("info", `sign-in waiting for browser authorization process=${p.process_id}`)
                }

                const remaining = deadline - Date.now()
                if (remaining <= 0) return { type: "failed", error: "The sign-in timed out; start it again" }
                await sleep(Math.min(pollEveryMs(), remaining), s)
              }
            },
          }
        },
      },
      {
        type: "api",
        label: "Vibe Code API key",
        placeholder: "key made in console.mistral.ai under Code > Vibe Code CLI",
      },
    ],
  },
  provider: {
    id: ID,
    async models(provider, { auth } = {}) {
      const key = keyOf(auth)
      if (!key) return provider?.models ?? {}
      const got = await listModels(key).catch(() => null)
      return got && Object.keys(got).length ? got : (provider?.models ?? {})
    },
  },
  }
}

// for tests
export const _internal = {
  startSignIn,
  pollSignIn,
  exchangeSignIn,
  whoami,
  listModels,
  readUsage,
  keyOf,
  priceOf,
  rateLimitWaitMs,
  userAgent,
  DOCS_PRICES,
  mistralBody,
  THINKING_FIELDS,
  ID,
  apiBase,
  consoleApi,
}
