/* 미팅 전사 — 오프라인 대응 서비스 워커.
 *
 * 원형: FED-Shell(github.com/git-fed/Build-Web-and-Mobile-Apps)의 sw.js
 * (오프라인 우선 PWA 셸 전략).
 *
 * 전략:
 *   - 앱 셸: 설치 시 선캐시, cache-first.
 *   - 버전 고정 CDN 자산: cache-first (불변 URL이라 안전).
 *   - 그 외 same-origin GET: stale-while-revalidate.
 *   - 내비게이션: network-first → 캐시 → offline.html.
 *
 * APP_VERSION은 앱 버전과 함께 올릴 것. 버전을 바꾸면 activate 단계에서
 * 이전 캐시가 자동 정리됨.
 */

const APP_VERSION = '1.0.1';
const SHELL_CACHE = `app-shell-${APP_VERSION}`;
const RUNTIME_CACHE = `app-runtime-${APP_VERSION}`;

const SHELL_ASSETS = [
  './',
  './index.html',
  './manifest.json',
  './offline.html',
  './icons/icon.svg',
  './icons/icon-maskable.svg',
];
const SHELL_ASSET_PATHS = new Set(
  SHELL_ASSETS.map((asset) => new URL(asset, self.registration.scope).pathname),
);

// 버전이 고정된 CDN 자산 (불변 → cache-first가 안전)
const CDN_PREFIXES = [];
const isCdnAsset = (url) => CDN_PREFIXES.some((prefix) => url.startsWith(prefix));

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(SHELL_CACHE)
      .then((cache) => cache.addAll(SHELL_ASSETS))
      .then(() => self.skipWaiting()),
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
