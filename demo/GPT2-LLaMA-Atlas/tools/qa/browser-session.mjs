// Shared lifecycle for local 3D QA. Never silently run WebGL on the CPU.
import net from 'node:net';
import { createRequire } from 'node:module';
import { setTimeout as delay } from 'node:timers/promises';

const LOCK_PORT = 4240;
const require = createRequire(import.meta.url);
const PLAYWRIGHT = '/Users/yunni/.npm/_npx/31e32ef8478fbf80/node_modules/playwright-core';

async function deadline(promise, ms, message) {
  let timer;
  try {
    return await Promise.race([promise, new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(message)), ms);
    })]);
  } finally { clearTimeout(timer); }
}

function listen(server, port) {
  return new Promise((resolve, reject) => {
    const fail = (error) => { server.off('listening', ready); reject(error); };
    const ready = () => { server.off('error', fail); resolve(); };
    server.once('error', fail);
    server.once('listening', ready);
    server.listen(port, '127.0.0.1');
  });
}

async function closeServer(server) {
  server.closeAllConnections?.();
  if (server.listening) await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
}

// Kernel-held lease: a crash/SIGKILL releases it without stale PID files.
export async function acquireBrowserLock({ port = LOCK_PORT, waitMs = 120_000, name = 'browser QA', onWait = console.log } = {}) {
  const expires = Date.now() + waitMs;
  let announced = false;
  while (true) {
    const server = net.createServer((socket) => {
      socket.on('error', () => {});
      socket.end('atlas-browser-qa\n');
    });
    try {
      await listen(server, port);
      return { port: server.address().port, close: () => closeServer(server) };
    } catch (error) {
      if (error.code !== 'EADDRINUSE') throw error;
      if (Date.now() >= expires) throw new Error(`等待浏览器测试锁超时：127.0.0.1:${port}`);
      if (!announced) { onWait(`[${name}] 等待另一组浏览器测试结束（锁端口 ${port}）`); announced = true; }
      await delay(Math.min(250, Math.max(1, expires - Date.now())));
    }
  }
}

export function requireMetal(renderer) {
  if (!renderer?.webgl2 || !/ANGLE Metal Renderer/i.test(renderer.renderer || '')
      || /swiftshader|llvmpipe|software/i.test(renderer.renderer || '')) {
    throw new Error(`3D 测试需要 Metal 硬件渲染，已停止，未回退软件渲染：${JSON.stringify(renderer)}`);
  }
}

async function inspectRenderer(browser) {
  const context = await browser.newContext({ viewport: { width: 32, height: 32 } });
  try {
    const page = await context.newPage();
    page.setDefaultTimeout(10_000);
    return await page.evaluate(() => {
      const canvas = document.createElement('canvas');
      canvas.width = canvas.height = 1;
      const gl = canvas.getContext('webgl2');
      if (!gl) return { webgl2: false };
      const extension = gl.getExtension('WEBGL_debug_renderer_info');
      const result = { webgl2: true, renderer: extension ? gl.getParameter(extension.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER) };
      gl.getExtension('WEBGL_lose_context')?.loseContext();
      return result;
    });
  } finally { await context.close(); }
}

async function stopBrowser(server, browser, graceMs) {
  if (!server) return;
  try {
    await deadline(server.close(), graceMs, 'Chrome 正常关闭超时');
  } catch {
    try { await deadline(server.kill(), graceMs, 'Chrome 强制关闭超时'); }
    catch (error) {
      // Playwright launches a detached process group on POSIX. Only kill our group.
      const child = server.process();
      if (child.exitCode == null && child.signalCode == null) {
        if (!Number.isInteger(child.pid) || child.pid <= 1 || child.pid === process.pid) throw error;
        try { process.kill(process.platform === 'win32' ? child.pid : -child.pid, 'SIGKILL'); }
        catch (killError) { if (killError.code !== 'ESRCH') throw killError; }
        await deadline(new Promise((resolve) => child.once('exit', resolve)), graceMs, error.message);
      }
    }
  } finally {
    if (browser) await deadline(browser.close(), graceMs, '断开 Chrome 连接超时').catch(() => {});
  }
}

export async function withBrowserSession(name, run, {
  timeoutMs = 180_000, lockPort = LOCK_PORT, lockWaitMs = 120_000,
  graceMs = 5_000, chromium: providedChromium,
} = {}) {
  const lease = await acquireBrowserLock({ port: lockPort, waitMs: lockWaitMs, name });
  const info = { name, startedAt: new Date().toISOString(), lockPort: lease.port, browserPid: null, renderer: null };
  const servers = new Set();
  let browserServer, browser, launching, stopped = false;
  let abort;
  const interrupted = new Promise((_, reject) => { abort = reject; });
  const onSignal = (signal) => abort(new Error(`浏览器测试被 ${signal} 中止`));
  const handlers = ['SIGINT', 'SIGTERM', 'SIGHUP'].map((signal) => {
    const handler = () => onSignal(signal);
    process.on(signal, handler);
    return [signal, handler];
  });
  const timer = setTimeout(() => abort(new Error(`浏览器测试超过 ${timeoutMs} ms，已中止`)), timeoutMs);
  const checkRunning = () => { if (stopped) throw new Error('浏览器测试已停止'); };
  const work = (async () => {
    const chromium = providedChromium || require(PLAYWRIGHT).chromium;
    launching = chromium.launchServer({
      channel: 'chrome', headless: true, timeout: 15_000,
      args: ['--allow-file-access-from-files'],
      handleSIGINT: false, handleSIGTERM: false, handleSIGHUP: false,
    });
    browserServer = await launching;
    info.browserPid = browserServer.process().pid;
    checkRunning();
    browser = await chromium.connect(browserServer.wsEndpoint(), { timeout: 10_000 });
    checkRunning();
    info.renderer = await inspectRenderer(browser);
    checkRunning();
    requireMetal(info.renderer);
    console.log(`[${name}] ${info.renderer.renderer} · PID ${info.browserPid}`);
    await run({ browser, info, listen: async (server, port) => {
      checkRunning();
      servers.add(server);
      await listen(server, port);
      checkRunning();
    } });
    return info;
  })();
  try {
    return await Promise.race([work, interrupted]);
  } finally {
    stopped = true;
    clearTimeout(timer);
    try {
      // Join any pending launch, so a timeout during startup cannot leak a late browser.
      if (!browserServer && launching) browserServer = await launching.catch(() => null);
      await stopBrowser(browserServer, browser, graceMs);
    } finally {
      try { await Promise.all([...servers].map(closeServer)); }
      finally {
        await lease.close();
        info.finishedAt = new Date().toISOString();
        for (const [signal, handler] of handlers) process.off(signal, handler);
      }
    }
  }
}
