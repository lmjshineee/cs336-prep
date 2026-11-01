// Real generation endings and temperature sampling, in isolated browser contexts.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { withBrowserSession } from './browser-session.mjs';
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const OUT = path.resolve(ROOT, '../output/playwright/generation-20261031');
fs.mkdirSync(OUT, { recursive: true });
const checks = [], errors = [];
function check(id, ok, detail) { checks.push({ id, passed: !!ok, detail }); if (!ok) console.error('FAIL', id, JSON.stringify(detail)); }
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json' };
const server = http.createServer((req, res) => {
  const file = path.join(ROOT, decodeURIComponent(req.url.split('?')[0]));
  fs.readFile(file, (error, data) => { res.writeHead(error ? 404 : 200, { 'Content-Type': types[path.extname(file)] || 'text/plain' }); res.end(error ? 'not found' : data); });
});
const models = [
  { key: 'gpt2', template: 'index.html', single: 'atlas.html', total: 145, generated: 9, text: 'The cat sat on the floor, and the cat was still asleep.' },
  { key: 'llama', template: 'llama.html', single: 'llama-atlas.html', total: 239, generated: 7, text: 'To make a cup of tea, first you need to boil the water.' },
];
async function seek(page, value) {
  await page.locator('#scrub').evaluate((el, v) => { el.value = String(v); el.dispatchEvent(new Event('input', { bubbles: true })); }, value);
}
let runtime = null, runtimeError = null;
try {
  await withBrowserSession('generation-ui', async ({ browser, info, listen }) => {
    runtime = info;
    await listen(server, 4241);
    for (const model of models) {
      for (const format of ['http', 'single-http', 'file']) {
        const id = `${model.key}.${format}`;
        const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
        context.setDefaultTimeout(10_000);
        context.setDefaultNavigationTimeout(15_000);
        await context.addInitScript(() => localStorage.setItem('gpt2llama.onboarded', '1'));
        const page = await context.newPage();
        const network = [];
        page.on('pageerror', (error) => errors.push(`${id}: ${error}`));
        page.on('console', (message) => { if (message.type() === 'error') errors.push(`${id}: ${message.text()}`); });
        if (format === 'file') page.on('request', (request) => { if (!request.url().startsWith('file:')) network.push(request.url()); });
        const url = format === 'file' ? `file://${path.join(ROOT, model.single)}` : `http://127.0.0.1:4241/${format === 'http' ? model.template : model.single}`;
        await page.goto(url);
        await page.waitForFunction(() => window.__trace?.().state === 'ready', null, { timeout: 30000 });
        check(`${id}.ready`, await page.evaluate((n) => window.__trace().total === n && window.__trace().shaVerified, model.total), await page.evaluate(() => window.__trace()));
        const before = await page.evaluate(() => [localStorage.getItem('gpt2.progress.v2'), localStorage.getItem('llama.progress.v1')]);
        check(`${id}.cycles`, (await page.locator('.cycle').allTextContents()).join(',') === ['P', ...Array.from({ length: model.generated - 1 }, (_, i) => String(i + 1))].join(','), await page.locator('.cycle').allTextContents());
        await seek(page, model.total - 1);
        check(`${id}.final_sentence`, (await page.locator('#sequence-text').textContent()) === model.text, await page.locator('#sequence-text').textContent());
        check(`${id}.no_pending`, await page.locator('#seq .is-pending').count() === 0, await page.locator('#seq').innerText());
        check(`${id}.stopped`, (await page.locator('#mode-chip').textContent()) === '完成' && (await page.locator('#input-meta').textContent()).includes('达到上限'), await page.locator('#input-meta').textContent());
        await seek(page, model.total - 2);
        check(`${id}.rewind`, await page.locator('#seq .is-pending').count() === 1, await page.locator('#mode-chip').textContent());
        await seek(page, model.total - 1);
        await page.locator('#btn-sampling').click();
        await page.locator('#sampling-temperature').waitFor();
        check(`${id}.round_default`, (await page.locator('#sampling-round').inputValue()) === String(model.generated - 1), await page.locator('#sampling-round').inputValue());
        await page.locator('#sampling-round').selectOption('0');
        const normal = await page.locator('#sampling-stats').textContent();
        await page.locator('[data-temperature="0.5"]').click();
        const cold = await page.locator('#sampling-stats').textContent();
        check(`${id}.temperature`, normal !== cold && (await page.locator('#sampling-column').textContent()) === 'T=0.5', { normal, cold });
        check(`${id}.full_vocabulary`, (await page.locator('#sampling-candidates tr').count()) === 9 && cold.includes(model.key === 'gpt2' ? '50,257' : '49,152'), cold);
        await page.locator('#sampling-draw').click();
        const first = await page.locator('#sampling-result').textContent();
        await page.locator('#sampling-reset').click();
        await page.locator('#sampling-draw').click();
        check(`${id}.seed_repeats`, first === await page.locator('#sampling-result').textContent(), first);
        await page.locator('#sampling-round').selectOption('1');
        check(`${id}.draw_cleared`, (await page.locator('#sampling-result').textContent()).includes('点击'), await page.locator('#sampling-result').textContent());
        if (format === 'http') {
          await page.locator('[data-temperature="1.5"]').click();
          await page.locator('#sampling-draw').click();
          await page.screenshot({ path: path.join(OUT, `${model.key}-temperature.png`) });
        }
        await page.locator('#modal-close').click();
        check(`${id}.original_unchanged`, (await page.locator('#sequence-text').textContent()) === model.text && (await page.evaluate(() => window.__realStep().kind)) === 'sample', await page.locator('#sequence-text').textContent());
        for (const width of format === 'http' ? [1440, 1200] : [1440]) {
          await page.setViewportSize({ width, height: 1000 });
          const layout = await page.evaluate(() => {
            const transport = document.querySelector('.transport').getBoundingClientRect();
            const foot = document.querySelector('.input-foot').getBoundingClientRect();
            const dock = document.querySelector('#dock').getBoundingClientRect();
            return { footBottom: foot.bottom, transportTop: transport.top, dockRight: dock.right, viewport: innerWidth, width: document.documentElement.scrollWidth };
          });
          check(`${id}.layout${width}`, layout.footBottom <= layout.transportTop + 1 && layout.dockRight <= width && layout.width <= width, layout);
        }
        if (format === 'http') await page.screenshot({ path: path.join(OUT, `${model.key}-final.png`) });
        const after = await page.evaluate(() => [localStorage.getItem('gpt2.progress.v2'), localStorage.getItem('llama.progress.v1')]);
        check(`${id}.no_learning_writes`, JSON.stringify(before) === JSON.stringify(after), { before, after });
        if (format === 'file') check(`${id}.offline`, network.length === 0, network);
        await context.close();
      }
    }
    check('no_browser_errors', errors.length === 0, errors);
  });
} catch (error) {
  runtimeError = error.stack || String(error);
  errors.push(runtimeError);
  console.error(runtimeError);
} finally {
  fs.writeFileSync(path.join(OUT, 'verification.json'), JSON.stringify({ runtime, completed: !runtimeError, checks, errors, passed: checks.filter((c) => c.passed).length, failed: checks.filter((c) => !c.passed).length + (runtimeError ? 1 : 0) }, null, 2));
}
console.log(JSON.stringify({ passed: checks.filter((c) => c.passed).length, failed: checks.filter((c) => !c.passed).length + (runtimeError ? 1 : 0), output: OUT }));
if (runtimeError || checks.some((c) => !c.passed) || errors.length) process.exitCode = 1;
