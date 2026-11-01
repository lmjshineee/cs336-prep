#!/usr/bin/env node
// LLaMA L2—L7 browser acceptance: HTTP page, single-file HTTP/file://, course, local viewer.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { withBrowserSession } from './browser-session.mjs';
import { LLAMA_LESSONS } from '../../js/llama-lessons.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..');
const SITE = ROOT;
const PROJECT = path.resolve(ROOT, '..');
const OUT = path.join(PROJECT, 'output', 'qa', 'llama-generation-20261031');
fs.mkdirSync(OUT, { recursive: true });
const checks = [];
function check(id, passed, observed, expected = true) {
  checks.push({ id, status: passed ? 'passed' : 'failed', observed, expected });
  if (!passed) console.error(`FAIL ${id}:`, JSON.stringify(observed).slice(0, 600));
}
const mime = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.py': 'text/plain; charset=utf-8', '.png': 'image/png' };
const server = http.createServer((req, res) => {
  const rel = req.url === '/' ? 'index.html' : req.url.split('?')[0];
  const file = path.join(SITE, rel);
  fs.readFile(file, (error, data) => {
    if (error) { res.writeHead(404); res.end('not found'); return; }
    res.writeHead(200, { 'Content-Type': mime[path.extname(file)] || 'application/octet-stream' });
    res.end(data);
  });
});
let browser, runtime = null, runtimeError = null;
const errors = [];
async function newPage() {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  context.setDefaultTimeout(10_000);
  context.setDefaultNavigationTimeout(15_000);
  await context.addInitScript(() => localStorage.setItem('gpt2llama.onboarded', '1'));
  const page = await context.newPage();
  page.on('pageerror', (error) => errors.push(String(error)));
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
  return { context, page };
}
async function waitTrace(page) {
  try {
    await page.waitForFunction(() => window.__trace && window.__trace().state === 'ready', null, { timeout: 30000 });
  } catch (error) {
    console.error('waitTrace failed', page.url(), errors.slice(0, 5), await page.evaluate(() => ({ trace: window.__trace?.(), entry: window.__entry, title: document.title })).catch((e) => String(e)));
    throw error;
  }
  return page.evaluate(() => window.__trace());
}

try {
  await withBrowserSession('llama-course-ui', async (session) => {
    browser = session.browser;
    runtime = session.info;
    await session.listen(server, 4238);
    // ---------------------------------------------------------------- HTTP page
    {
      const { context, page } = await newPage();
      await page.goto('http://127.0.0.1:4238/llama.html');
      const trace = await waitTrace(page);
      check('http.real_ready', trace.mode === 'real' && trace.total === 239 && trace.model === 'SmolLM2-135M', trace);
      check('http.brand', (await page.locator('.brand h1').textContent()) === 'Llama 架构', await page.locator('.brand h1').textContent());
      check('http.model_name', (await page.locator('#sel-name').textContent()) === 'SmolLM2-135M', await page.locator('#sel-name').textContent());
      const sceneNodes = await page.evaluate(() => document.querySelectorAll('#tree .trow').length);
      check('http.tree_nodes', sceneNodes > 30, sceneNodes);
      const oldButton = page.locator('#btn-legacy-llama2');
      check('http.legacy_entry', await oldButton.count() === 1, await oldButton.count());
      await oldButton.click();
      await page.waitForTimeout(200);
      const legacyText = await page.locator('#modal-body').innerText();
      check('http.legacy_modal_identity', legacyText.includes('旧 LLaMA-2') && legacyText.includes('不是当前 SmolLM2'), legacyText.slice(0, 120));
      await page.locator('#modal-close').click();
      await page.locator('#btn-sources').click();
      await page.waitForTimeout(150);
      const sourceText = await page.locator('#modal-body').innerText();
      check('http.source_modal_identity', sourceText.includes('SmolLM2-135M 主实例') && sourceText.includes('smollm2'), sourceText.slice(0, 160));
      await page.locator('#modal-close').click();

      // L2 seek/back: sample then block, cache state from events
      await page.evaluate(() => { const range = document.querySelector('#scrub'); range.value = '2'; range.dispatchEvent(new Event('input', { bubbles: true })); });
      await page.waitForTimeout(200);
      const sampleState = await page.evaluate(() => ({ ...window.__lesson(), trace: window.__trace(), step: window.__trace() }));
      check('http.seek_step', sampleState.trace.total === 239 && sampleState.trace.steps === 239, { total: sampleState.trace.total, steps: sampleState.trace.steps });
      const cacheText = await page.locator('#cache-strip').innerText().catch(() => '');
      check('http.cache_visible', cacheText.length > 0, cacheText.slice(0, 80));

      // L4 local viewer
      await page.locator('#btn-real-data').click();
      await page.waitForFunction(() => window.__llamaLocal && window.__llamaLocal().sourceStatus === 'ready', null, { timeout: 20000 });
      const attention = await page.evaluate(() => window.__llamaLocal());
      check('viewer.source_verified', attention.sourceVerified === true, attention.sourceVerified);
      check('viewer.attention_rows', attention.detail && attention.detail.rowSumsError < 1e-6, attention.detail);
      await page.locator('.lv-tab[data-ltab="rope"]').click();
      await page.waitForTimeout(150);
      const rope = await page.evaluate(() => window.__llamaLocal());
      check('viewer.rope_recompute', rope.detail && rope.detail.qError < 1e-5 && rope.detail.kError < 1e-5, rope.detail);
      await page.locator('.lv-tab[data-ltab="rmsnorm"]').click();
      await page.waitForTimeout(150);
      const rms = await page.evaluate(() => window.__llamaLocal());
      check('viewer.rms_recompute', rms.detail && rms.detail.maxErr < 1e-5, rms.detail);
      await page.locator('.lv-tab[data-ltab="swiglu"]').click();
      await page.waitForTimeout(150);
      const swiglu = await page.evaluate(() => window.__llamaLocal());
      check('viewer.swiglu_recompute', swiglu.detail && swiglu.detail.maxErr < 1e-6, swiglu.detail);
      await page.locator('.lv-tab[data-ltab="gqa"]').click();
      await page.waitForTimeout(150);
      const gqa = await page.evaluate(() => window.__llamaLocal());
      check('viewer.gqa_mapping', gqa.detail && JSON.stringify(gqa.detail.mapping) === '[0,0,0,1,1,1,2,2,2]', gqa.detail?.mapping);
      await page.locator('.lv-tab[data-ltab="attention"]').click();
      await page.selectOption('[data-lv="层"]', '4');
      await page.waitForTimeout(120);
      await page.locator('.lv-tab[data-ltab="rmsnorm"]').click();
      await page.waitForTimeout(120);
      const layer4 = await page.evaluate(() => window.__llamaLocal());
      check('viewer.layer_boundary', !!(layer4.detail && layer4.detail.error), layer4.detail);
      await page.locator('#lv-close').click();

      // L5 all nine lessons and all 40 steps; answers are taken from the course data.
      check('course.count', await page.locator('.course').count() === 9, await page.locator('.course').count());
      check('course.steps', LLAMA_LESSONS.reduce((n, lesson) => n + lesson.steps.length, 0) === 40, LLAMA_LESSONS.reduce((n, lesson) => n + lesson.steps.length, 0));
      for (let li = 0; li < LLAMA_LESSONS.length; li += 1) {
        const lesson = LLAMA_LESSONS[li];
        await page.locator(`.course[data-i="${li}"]`).click();
        await page.waitForTimeout(80);
        let guard = 0;
        while (!(await page.evaluate(() => window.__lesson().quizVisible)) && guard <= lesson.steps.length + 1) {
          await page.locator('#lb-next').click();
          await page.waitForTimeout(25);
          guard += 1;
        }
        const quiz = await page.evaluate(() => window.__lesson());
        check(`course.${lesson.id}.quiz_visible`, quiz.quizVisible === true, quiz.quizVisible);
        const stateStep = await page.evaluate(() => window.__lesson().step);
        check(`course.${lesson.id}.step_reached`, stateStep === lesson.steps.length - 1, { stateStep, expected: lesson.steps.length - 1 });
        await page.locator('.lb-opt').nth(lesson.quiz.answer).click();
        await page.waitForTimeout(80);
        const after = await page.evaluate(() => window.__progress());
        check(`course.${lesson.id}.quiz_passed`, after.lessons?.[lesson.id]?.quizPassed === true, after.lessons?.[lesson.id]);
        await page.locator('#lb-next').click();
        await page.waitForTimeout(100);
      }
      const finalProgress = await page.evaluate(() => window.__progress());
      check('course.all_passed', Object.values(finalProgress.lessons).filter((x) => x.quizPassed).length === 9, Object.values(finalProgress.lessons).filter((x) => x.quizPassed).length);
      check('course.storage_key', finalProgress.key === 'llama.progress.v1' && finalProgress.model === 'smollm2' && finalProgress.revision === 'llama-r1', { key: finalProgress.key, model: finalProgress.model, revision: finalProgress.revision });
      await page.reload();
      await waitTrace(page);
      const reloaded = await page.evaluate(() => window.__progress());
      check('course.reload_persist', Object.values(reloaded.lessons).filter((x) => x.quizPassed).length === 9, Object.values(reloaded.lessons).filter((x) => x.quizPassed).length);
      check('course.gpt2_keys_untouched', await page.evaluate(() => localStorage.getItem('gpt2.progress.v2') === null && localStorage.getItem('gpt2llama.lessons') === null), await page.evaluate(() => ({ gpt2: localStorage.getItem('gpt2.progress.v2'), legacy: localStorage.getItem('gpt2llama.lessons') })));
      await page.screenshot({ path: path.join(OUT, 'http-course.png') });
      await context.close();
    }

    // ------------------------------------------------------- HTTP single file
    {
      const { context, page } = await newPage();
      await page.goto('http://127.0.0.1:4238/llama-atlas.html');
      const trace = await waitTrace(page);
      check('single_http.real_ready', trace.mode === 'real' && trace.total === 239, trace);
      await page.locator('#btn-real-data').click();
      await page.waitForFunction(() => window.__llamaLocal && window.__llamaLocal().sourceStatus === 'ready', null, { timeout: 20000 });
      check('single_http.source_verified', (await page.evaluate(() => window.__llamaLocal())).sourceVerified === true, true);
      await context.close();
    }

    // -------------------------------------------- file:// single file + copied dir
    const requests = [];
    for (const [id, target] of [['file.single', path.join(SITE, 'llama-atlas.html')], ['file.copied', path.join(OUT, 'llama-atlas.html')]]) {
      if (id === 'file.copied') fs.copyFileSync(path.join(SITE, 'llama-atlas.html'), target);
      const { context, page } = await newPage();
      page.on('request', (request) => { if (!request.url().startsWith('file:')) requests.push(request.url()); });
      await page.goto(`file://${target}`);
      const trace = await waitTrace(page);
      check(`${id}.ready`, trace.mode === 'real' && trace.total === 239, trace);
      check(`${id}.sha_verified`, trace.shaVerified === true, trace.shaVerified);
      await page.locator('#btn-real-data').click();
      await page.waitForFunction(() => window.__llamaLocal && window.__llamaLocal().sourceStatus === 'ready', null, { timeout: 20000 });
      const viewer = await page.evaluate(() => window.__llamaLocal());
      check(`${id}.source_verified`, viewer.sourceVerified === true, viewer.sourceVerified);
      check(`${id}.no_network`, requests.length === 0, requests.slice(0, 5));
      await page.screenshot({ path: path.join(OUT, `${id}.png`) });
      await context.close();
    }

    // --------------------------------------------------------------- GPT-2 file
    {
      const { context, page } = await newPage();
      await page.goto(`file://${path.join(SITE, 'atlas.html')}`);
      const trace = await waitTrace(page);
      check('gpt2.file_ready', trace.mode === 'real' && trace.total === 145, trace);
      await context.close();
    }

  });
} catch (error) {
  runtimeError = error.stack || String(error);
  errors.push(runtimeError);
  console.error(runtimeError);
} finally {
  const failed = checks.filter((c) => c.status === 'failed');
  const result = {
    generated_at: new Date().toISOString(), runtime, completed: !runtimeError,
    checks: checks.length, passed: checks.length - failed.length,
    failed: failed.length + (runtimeError ? 1 : 0),
    failures: failed, page_errors: errors, checks_detail: checks,
  };
  fs.writeFileSync(path.join(OUT, 'ui-checks.json'), `${JSON.stringify(result, null, 2)}\n`);
  console.log(`checks ${result.passed}/${result.checks}; page_errors=${errors.length}; failures=${result.failed}`);
  if (result.failed || errors.length) process.exitCode = 1;
}
