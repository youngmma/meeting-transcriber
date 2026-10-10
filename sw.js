/* Meeting Transcriber — offline service worker.
 *
 * Based on FED-Shell (github.com/git-fed/Build-Web-and-Mobile-Apps) sw.js
 * (offline-first PWA shell strategy).
 *
 * Strategy:
 *   - app shell: precache on install, cache-first.
 *   - version-pinned CDN assets: cache-first (immutable URLs, safe).
 *   - other same-origin GETs: stale-while-revalidate.
 *   - navigation: network-first → cache → offline.html.
 *
 * Bump APP_VERSION together with the app version. Changing it clears the
 * old cache during activate.
 */

const APP_VERSION = '1.0.85';
const SHELL_CACHE = `app-shell-${APP_VERSION}`;
const RUNTIME_CACHE = `app-runtime-${APP_VERSION}`;

const SHELL_ASSETS = [
  './',
  './index.html',
  './whisper-worker.js',
  './manifest.json',
  './offline.html',
  './icons/icon.svg',
  './icons/icon-maskable.svg',
];
const SHELL_ASSET_PATHS = new Set(
  SHELL_ASSETS.map((asset) => new URL(asset, self.registration.scope).pathname),
);

// version-pinned CDN assets (immutable → cache-first is safe)
// UAT-35: the transformers runtime library must work offline — without this,
// airplane mode breaks transcription even when models are downloaded.
const CDN_PREFIXES = ['https://cdn.jsdelivr.net/npm/@huggingface/transformers@'];
const isCdnAsset = (url) => CDN_PREFIXES.some((prefix) => url.startsWith(prefix));

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(SHELL_CACHE)
      .then((cache) => cache.addAll(SHELL_ASSETS)),
    // LC-10: do NOT call skipWaiting() here. A new worker activating mid-session
    // can invalidate caches the running page depends on. Activation happens only
    // when the user clicks "Update now" (page posts {type:'skipWaiting'}), which
    // is itself suppressed during active recording/transcription (LC-05).
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((k) => k.startsWith('app-') && k !== SHELL_CACHE && k !== RUNTIME_CACHE)
            .map((k) => caches.delete(k)),
        ),
      )
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);

  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request)
        .then((response) => {
          event.waitUntil(
            caches
              .open(RUNTIME_CACHE)
              .then((cache) => cache.put(request, response.clone()))
              .catch(() => {}),
          );
          return response;
        })
        .catch(async () => {
          const cached = await caches.match(request);
          return cached || (await caches.match('./offline.html'));
        }),
    );
    return;
  }

  if (url.origin === self.location.origin && SHELL_ASSET_PATHS.has(url.pathname)) {
    event.respondWith(caches.match(request).then((cached) => cached || fetch(request)));
    return;
  }

  if (isCdnAsset(request.url)) {
    event.respondWith(
      caches.match(request).then(
        (cached) =>
          cached ||
          fetch(request).then((response) => {
            if (response && response.status === 200 && response.type !== 'opaque') {
              event.waitUntil(
                caches
                  .open(RUNTIME_CACHE)
                  .then((cache) => cache.put(request, response.clone()))
                  .catch(() => {}),
              );
            }
            return response;
          }),
      ),
    );
    return;
  }

  if (url.origin === self.location.origin) {
    event.respondWith(
      caches.match(request).then((cached) => {
        const fetchPromise = fetch(request)
          .then((response) => {
            event.waitUntil(
              caches
                .open(RUNTIME_CACHE)
                .then((cache) => cache.put(request, response.clone()))
                .catch(() => {}),
            );
            return response;
          })
          .catch(() => cached);
        return cached || fetchPromise;
      }),
    );
  }
});

self.addEventListener('message', (event) => {
  if (event.data === 'skipWaiting') self.skipWaiting();
});
