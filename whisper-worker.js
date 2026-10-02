/* 미팅 전사 — Whisper 전사 전용 Web Worker.
 *
 * 메인 스레드 멈춤("This page isn't responding") 방지를 위해
 * 무거운 ONNX 추론을 워커로 분리. 30초 이하의 PCM 윈도우 하나를
 * 받아 전사한 뒤 타임스탬프 청크 배열을 반환한다.
 * 모델 파일은 페이지와 같은 Cache API 캐시(cacheKey)를 공유한다.
 */
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0/dist/transformers.min.js';

env.cacheKey = 'meeting-transcriber-whisper';

let asr = null;
let modelId = null;

self.onmessage = async (e) => {
  const m = e.data || {};
  try {
    if (m.type === 'init') {
      if (asr && modelId === m.model) { self.postMessage({ type: 'ready' }); return; }
      modelId = m.model;
      asr = await pipeline('automatic-speech-recognition', modelId, {
        device: 'wasm',
        dtype: 'q8',
        progress_callback: (p) => self.postMessage({
          type: 'modelProgress',
          file: p.file || '', loaded: p.loaded || 0, total: p.total || 0,
        }),
      });
      self.postMessage({ type: 'ready' });
    } else if (m.type === 'transcribe') {
      if (!asr) throw new Error('모델이 초기화되지 않았습니다');
      const out = await asr(m.pcm, {
        task: 'transcribe',
        return_timestamps: true,
        chunk_length_s: 30,
        stride_length_s: 0,
        // 반복 환각 방지: 같은 3-gram이 두 번 나오면 금지 ("two types of" 무한루프 차단)
        no_repeat_ngram_size: 3,
        // 30초 발화에 224 토큰이면 충분. 폭주 루프의 토큰 낭비(시간 낭비) 차단
        max_new_tokens: 224,
      });
      self.postMessage({ type: 'done', id: m.id, chunks: out.chunks || [] });
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, stage: m.type,
      message: String((err && err.message) || err).slice(0, 300) });
  }
};
