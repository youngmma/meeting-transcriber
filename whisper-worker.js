/* Meeting Transcriber — Whisper-only Web Worker.
 *
 * Split heavy ONNX inference into a worker to avoid
 * main-thread freezes ("This page isn't responding"). Takes one
 * PCM window (<=30s), transcribes it, and returns timestamped chunks.
 * Model files share the page's Cache API cache (cacheKey).
 */
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0/dist/transformers.min.js';

env.cacheKey = 'meeting-transcriber-whisper';

let asr = null;
let modelId = null;

/** Detect language from 30s of PCM → 2-letter code like 'ko'/'en', by reading whisper's language token directly */
async function detectLanguage(pcm){
  const processed = await asr.processor(pcm);
  const startId = asr.model.config.decoder_start_token_id;
  const out = await asr.model.generate({
    inputs: processed.input_features,
    decoder_input_ids: [[startId]],
    max_new_tokens: 1,
  });
  const tokens = out[0].tolist(); // [startoftranscript, lang_token]
  const langId = Number(tokens[1]); // tolist() returns BigInts → must convert to Number before comparing
  const langMap = (asr.model.generation_config && asr.model.generation_config.lang_to_id) || {};
  for (const tok in langMap){
    if (langMap[tok] === langId){
      const m = /<\|([a-z]{2})\|>/.exec(tok);
      if (m) return m[1];
    }
  }
  return 'en';
}

self.onmessage = async (e) => {
  const m = e.data || {};
  try {
    if (m.type === 'init') {
      if (asr && modelId === m.model) { self.postMessage({ type: 'ready' }); return; }
      modelId = m.model;
      // Prefer WebGPU when available (much faster on GPUs, incl. integrated);
      // fall back to WASM. WebGPU needs fp32 weights (q8 is WASM-only).
      let device = 'wasm', dtype = 'q8';
      // For turbo, prefer q4 for smaller download (~400MB vs 656MB)
      const isTurboModel = /whisper-large-v3-turbo/.test(modelId);
      try {
        if (typeof navigator !== 'undefined' && navigator.gpu && !isTurboModel) {
          const adapter = await navigator.gpu.requestAdapter();
          // fp32 needs ~4x memory (small: 1GB). Use q8 on low-memory devices.
          const devMem = (typeof navigator.deviceMemory === 'number') ? navigator.deviceMemory : 8;
          if (adapter && !isTurboModel) {
            device = 'webgpu';
            dtype = devMem >= 8 ? 'fp32' : 'q8';
          } else if (isTurboModel) {
            // Turbo: use WASM q4 for smaller size
            device = 'wasm';
            dtype = 'q4';
          }
        }
      } catch(_) { device = 'wasm'; dtype = 'q8'; }
      const progCb = (p) => self.postMessage({
        type: 'modelProgress',
        file: p.file || '', loaded: p.loaded || 0, total: p.total || 0,
      });
      const tryLoad = (dev, dt) => pipeline('automatic-speech-recognition', modelId, {
        device: dev, dtype: dt, progress_callback: progCb,
      });
      try {
        asr = await tryLoad(device, dtype);
      } catch(e) {
        const isOOM = /bad_alloc|out of memory|memory/i.test(String((e && e.message) || e));
        const isSmallMed = /whisper-(small|medium)/.test(modelId);
        const isTurbo = /whisper-large-v3-turbo/.test(modelId);
        // Chain: WebGPU fp32 → WebGPU fp16 → WASM q8 → WASM q4
        if (device === 'webgpu' && dtype === 'fp32' && isSmallMed) {
          self.postMessage({ type: 'modelProgress', file: 'fp32 failed, trying fp16', loaded: 0, total: 1 });
          try { asr = await tryLoad('webgpu', 'fp16'); device = 'webgpu'; dtype = 'fp16';
          } catch(e2) { asr = null; }
        }
        if (!asr && isOOM && (isSmallMed || isTurbo) && dtype !== 'q4') {
          self.postMessage({ type: 'modelProgress', file: 'retrying with q4 (smaller)', loaded: 0, total: 1 });
          try { asr = await tryLoad(device === 'webgpu' ? 'webgpu' : 'wasm', 'q4'); dtype = 'q4';
            self.postMessage({ type: 'q4fallback' });
          } catch(e3) { asr = null; }
        }
        if (!asr) {
          if (device !== 'wasm') {
            // Last resort: WASM q8
            asr = await tryLoad('wasm', 'q8');
          } else throw e;
        }
        // P2: If small/medium/turbo still fails with OOM, fallback to base
        if (!asr && isOOM && /whisper-(small|medium|large)/.test(modelId)){
          self.postMessage({ type: 'modelProgress', file: 'falling back to base model', loaded: 0, total: 1 });
          const baseId = 'Xenova/whisper-base';
          asr = await pipeline('automatic-speech-recognition', baseId, {
            device: 'wasm', dtype: 'q8', progress_callback: progCb,
          });
          modelId = baseId;
          self.postMessage({ type: 'baseFallback' });
        }
      }
      self.postMessage({ type: 'ready', device, dtype, model: modelId });
    } else if (m.type === 'detect') {
      if (!asr) throw new Error('model not initialized');
      const language = await detectLanguage(m.pcm);
      self.postMessage({ type: 'detected', id: m.id, language });
    } else if (m.type === 'transcribe') {
      if (!asr) throw new Error('model not initialized');
      // Domain prompts: user's main contexts (work + church, EN + KO)
      // NOTE (UAT 2026-10-08): transformers.js 4.3.0 silently ignores the
      // `initial_prompt` string option, so this is currently a no-op placeholder.
      // A manual decoder_input_ids implementation was validated and REVERTED:
      // it made whisper-base output garbage ("다리, 기원, 기웅, 기" vs good
      // baseline). Kept as documentation of intent; harmless.
      const EN_PROMPT = 'State Department meeting, passport modernization, AIFM, TDIS, OCCAM pilot. Church sermon, Bible study, prayer meeting, David, Moses, Abraham, Jesus Christ, God, Holy Spirit.';
      const KO_PROMPT = '한국어 교회 설교, 성경 공부, 예배, 기도회, 순모임. 다윗, 모세, 아브라함, 예수님, 하나님, 성령님. 여호와, 이스라엘, 예루살렘.';
      const prompt = m.language === 'ko' ? KO_PROMPT
        : m.language === 'en' ? EN_PROMPT
        : KO_PROMPT + ' ' + EN_PROMPT; // auto-detect: include both
      const out = await asr(m.pcm, {
        language: m.language || undefined,
        task: 'transcribe',
        return_timestamps: true,
        chunk_length_s: 30,
        stride_length_s: 0,
        // lower temperature = more deterministic, fewer hallucinations
        temperature: 0.2,
        ...(prompt ? { initial_prompt: prompt } : {}),
        // anti-hallucination: ban any 3-gram appearing twice (blocks "two types of" infinite loops)
        no_repeat_ngram_size: 3,
        // 224 tokens is plenty for 30s of speech; blocks token/time waste from runaway loops
        max_new_tokens: 224,
      });
      self.postMessage({ type: 'done', id: m.id, chunks: out.chunks || [] });
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, stage: m.type,
      message: String((err && err.message) || err).slice(0, 300) });
  }
};
