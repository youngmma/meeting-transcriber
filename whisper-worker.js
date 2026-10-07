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
      try {
        if (typeof navigator !== 'undefined' && navigator.gpu) {
          const adapter = await navigator.gpu.requestAdapter();
          if (adapter) { device = 'webgpu'; dtype = 'fp32'; }
        }
      } catch(_) { device = 'wasm'; dtype = 'q8'; }
      try {
        asr = await pipeline('automatic-speech-recognition', modelId, {
          device, dtype,
          progress_callback: (p) => self.postMessage({
            type: 'modelProgress',
            file: p.file || '', loaded: p.loaded || 0, total: p.total || 0,
          }),
        });
      } catch(e) {
        if (device !== 'wasm') {
          // WebGPU failed (e.g. shader compile) → retry on WASM
          asr = await pipeline('automatic-speech-recognition', modelId, {
            device: 'wasm', dtype: 'q8',
            progress_callback: (p) => self.postMessage({
              type: 'modelProgress',
              file: p.file || '', loaded: p.loaded || 0, total: p.total || 0,
            }),
          });
        } else throw e;
      }
      self.postMessage({ type: 'ready', device });
    } else if (m.type === 'detect') {
      if (!asr) throw new Error('model not initialized');
      const language = await detectLanguage(m.pcm);
      self.postMessage({ type: 'detected', id: m.id, language });
    } else if (m.type === 'transcribe') {
      if (!asr) throw new Error('model not initialized');
      const out = await asr(m.pcm, {
        language: m.language || undefined,
        task: 'transcribe',
        return_timestamps: true,
        chunk_length_s: 30,
        stride_length_s: 0,
        // lower temperature = more deterministic, fewer hallucinations
        temperature: 0.2,
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
