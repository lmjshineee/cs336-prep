// Lifecycle checks use fake browsers. They do not load Chrome or draw 3D scenes.
import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { setTimeout as delay } from 'node:timers/promises';
import { acquireBrowserLock, requireMetal, withBrowserSession } from './browser-session.mjs';

const metal = { webgl2: true, renderer: 'ANGLE (Apple, ANGLE Metal Renderer: Apple M5, Unspecified Version)' };
function fakeChromium({ events = [], launchDelay = 0, hangClose = false, renderer = metal } = {}) {
  const browser = {
    newContext: async () => ({
      newPage: async () => ({ setDefaultTimeout() {}, evaluate: async () => renderer }),
      close: async () => events.push('probe-context-closed'),
    }),
    close: async () => events.push('connection-closed'),
  };
  return {
    async launchServer(options) {
      assert.equal(options.channel, 'chrome');
      assert.equal(options.executablePath, undefined);
      assert.ok(options.args.every((arg) => !/swiftshader/i.test(arg)));
      await delay(launchDelay);
      events.push('launched');
      return {
        process: () => ({ pid: 0 }), // Never a real process in these tests.
        wsEndpoint: () => 'fake-browser',
        close: async () => { events.push('closing'); if (hangClose) await new Promise(() => {}); events.push('closed'); },
        kill: async () => events.push('killed'),
      };
    },
    connect: async () => browser,
  };
}

test('Metal is required; software, missing WebGL and unidentified renderers are rejected', () => {
  assert.doesNotThrow(() => requireMetal(metal));
  for (const renderer of [
    { webgl2: false },
    { webgl2: true, renderer: 'ANGLE (SwiftShader driver)' },
    { webgl2: true, renderer: 'llvmpipe' },
    { webgl2: true, renderer: 'WebKit WebGL' },
  ]) assert.throws(() => requireMetal(renderer), /未回退软件渲染/);
});

test('independent clients cannot hold the browser lock simultaneously', async () => {
  const first = await acquireBrowserLock({ port: 0 });
  let acquired = false;
  const waiting = [];
  const pending = acquireBrowserLock({ port: first.port, waitMs: 2000, onWait: (message) => waiting.push(message) })
    .then((lease) => { acquired = true; return lease; });
  await delay(30);
  assert.equal(acquired, false);
  assert.equal(waiting.length, 1);
  await first.close();
  const second = await pending;
  assert.equal(acquired, true);
  await second.close();
});

test('a busy lock has a bounded wait and never steals another run’s lease', async () => {
  const lease = await acquireBrowserLock({ port: 0 });
  try {
    await assert.rejects(acquireBrowserLock({ port: lease.port, waitMs: 20, onWait() {} }), /等待浏览器测试锁超时/);
  } finally { await lease.close(); }
});

test('an assertion failure closes Chrome and HTTP server before releasing the lease', async () => {
  const events = [];
  const server = http.createServer((_req, res) => res.end('test'));
  let port;
  await assert.rejects(withBrowserSession('failure-test', async (session) => {
    port = session.info.lockPort;
    await session.listen(server, 0);
    throw new Error('deliberate assertion failure');
  }, { lockPort: 0, chromium: fakeChromium({ events }) }), /deliberate assertion failure/);
  assert.equal(server.listening, false);
  assert.ok(events.includes('closed'));
  const lease = await acquireBrowserLock({ port, waitMs: 20 });
  await lease.close();
});

test('software rendering closes the browser before the 3D suite can run', async () => {
  const events = [];
  await assert.rejects(withBrowserSession('software-test', async () => events.push('suite-ran'), {
    lockPort: 0,
    chromium: fakeChromium({ events, renderer: { webgl2: true, renderer: 'ANGLE (SwiftShader driver)' } }),
  }), /未回退软件渲染/);
  assert.ok(events.includes('closed'));
  assert.ok(!events.includes('suite-ran'));
});

test('an overall timeout cleans up even if the test callback never resolves', async () => {
  const events = [];
  const server = http.createServer();
  await assert.rejects(withBrowserSession('timeout-test', async (session) => {
    await session.listen(server, 0);
    await new Promise(() => {});
  }, { lockPort: 0, timeoutMs: 50, chromium: fakeChromium({ events }) }), /超过 50 ms/);
  assert.equal(server.listening, false);
  assert.ok(events.includes('closed'));
});

test('a timeout during launch joins and closes the late browser without running the suite', async () => {
  const events = [];
  await assert.rejects(withBrowserSession('late-launch-test', async () => events.push('suite-ran'), {
    lockPort: 0, timeoutMs: 10, chromium: fakeChromium({ events, launchDelay: 40 }),
  }), /超过 10 ms/);
  assert.ok(events.includes('closed'));
  assert.ok(!events.includes('suite-ran'));
});

test('a stalled graceful close escalates to the owned browser kill API', async () => {
  const events = [];
  await withBrowserSession('stalled-close-test', async () => {}, {
    lockPort: 0, graceMs: 10, chromium: fakeChromium({ events, hangClose: true }),
  });
  assert.ok(events.includes('killed'));
  assert.ok(events.includes('connection-closed'));
});

async function childWithFirstJson(source) {
  const child = spawn(process.execPath, ['--input-type=module', '-e', source], { stdio: ['ignore', 'pipe', 'pipe'] });
  let output = '', errors = '';
  child.stderr.on('data', (data) => { errors += data; });
  const line = await new Promise((resolve, reject) => {
    child.stdout.on('data', (data) => {
      output += data;
      const lines = output.split('\n'); lines.pop();
      const json = lines.find((text) => text.startsWith('{'));
      if (json) resolve(json);
    });
    child.once('error', reject);
    child.once('exit', (code) => { if (!output.includes('\n')) reject(new Error(`child ${code}: ${errors}`)); });
  });
  return { child, line, output: () => output };
}

test('the kernel releases the lease when its owner is killed', { timeout: 5000 }, async () => {
  const moduleUrl = new URL('./browser-session.mjs', import.meta.url).href;
  const { child, line } = await childWithFirstJson(`
    import { acquireBrowserLock } from ${JSON.stringify(moduleUrl)};
    const lease = await acquireBrowserLock({port: 0});
    console.log(JSON.stringify({port: lease.port}));
  `);
  const { port } = JSON.parse(line);
  const exited = once(child, 'exit');
  child.kill('SIGKILL');
  await exited;
  const lease = await acquireBrowserLock({ port, waitMs: 100 });
  await lease.close();
});

test('SIGTERM takes the same cleanup path and releases the shared lease', { timeout: 5000 }, async () => {
  const moduleUrl = new URL('./browser-session.mjs', import.meta.url).href;
  const { child, line, output } = await childWithFirstJson(`
    import { withBrowserSession } from ${JSON.stringify(moduleUrl)};
    const renderer = ${JSON.stringify(metal)};
    const chromium = {
      launchServer: async () => ({process:()=>({pid:0}),wsEndpoint:()=>'',close:async()=>console.log('browser-closed'),kill:async()=>{}}),
      connect: async () => ({close:async()=>{},newContext:async()=>({close:async()=>{},newPage:async()=>({setDefaultTimeout(){},evaluate:async()=>renderer})})})
    };
    try { await withBrowserSession('signal-test', async ({info}) => {
      console.log(JSON.stringify({port:info.lockPort})); await new Promise(()=>{});
    }, {lockPort:0,chromium}); } catch(e) {console.log(e.message);process.exitCode=1;}
  `);
  const { port } = JSON.parse(line);
  const exited = once(child, 'exit');
  child.kill('SIGTERM');
  const [code] = await exited;
  assert.equal(code, 1);
  assert.match(output(), /browser-closed/);
  assert.match(output(), /SIGTERM/);
  const lease = await acquireBrowserLock({ port, waitMs: 100 });
  await lease.close();
});
