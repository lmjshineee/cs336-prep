// 生成可双击打开的单文件版本：GPT-2（atlas.html）与 SmolLM2 Llama（llama-atlas.html）
//
// 用法：
//   npm i --no-save esbuild
//   node tools/build-standalone.mjs
//
// 每个入口模板共用同一份 js/main.js 打包结果；入口差异（window.ATLAS_ENTRY）
// 写在各自的 HTML 里，构建时原样保留。CSS 与 Three.js 全部内联，
// 结果不依赖网络、不需要服务器，file:// 直接打开即可运行。

import * as esbuild from 'esbuild';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const vendor = path.join(root, 'vendor');

const ENTRIES = [
  { template: 'index.html', output: 'atlas.html', title: 'GPT-2 · 解码器架构学习图谱', label: 'GPT-2', inlineData: true },
  { template: 'llama.html', output: 'llama-atlas.html', title: 'Llama 架构 · SmolLM2-135M 学习图谱', label: 'SmolLM2', inlineLlamaData: true },
];

// S7: put the verified trace + source snapshot into the GPT-2 single file. The loaders
// already prefer window.__ATLAS_TRACE_BUNDLE__ / __ATLAS_SOURCE_BUNDLE__, so no network
// request is made from file://. The raw case text is embedded so the browser can re-hash it.
function inlineScriptJson(value) {
  return JSON.stringify(value)
    .replace(/</g, '\\u003c')
    .replace(/\u2028/g, '\\u2028')
    .replace(/\u2029/g, '\\u2029');
}

let tracePrelude = null;
function getTracePrelude() {
  if (tracePrelude != null) return tracePrelude;
  const dataDir = path.join(root, 'data', 'gpt2');
  const sourceDir = path.join(root, 'sources', 'gpt2');
  const manifest = JSON.parse(readFileSync(path.join(dataDir, 'manifest.json'), 'utf8'));
  const caseText = readFileSync(path.join(dataDir, 'gpt2-main-002.json'), 'utf8');
  const caseData = JSON.parse(caseText);
  const caseSha256 = createHash('sha256').update(caseText, 'utf8').digest('hex');
  const instance = (manifest.instances || []).find((item) => item.case_id === 'gpt2-main-002');
  if (!instance) throw new Error('清单缺少 gpt2-main-002');
  if (instance.case_sha256 !== caseSha256) throw new Error('内联案例 SHA-256 与清单不一致');
  if (caseData.schema !== 'atlas.trace.case/1' || caseData.case_id !== 'gpt2-main-002') {
    throw new Error('内联案例格式/ID 不符合数据契约');
  }
  const verification = JSON.parse(readFileSync(path.join(dataDir, 'verification.json'), 'utf8'));
  const map = JSON.parse(readFileSync(path.join(sourceDir, 'source-map.json'), 'utf8'));
  const source = readFileSync(path.join(sourceDir, 'gpt2_real.py'), 'utf8');
  const sourceSha256 = createHash('sha256').update(source, 'utf8').digest('hex');
  if (sourceSha256 !== map.source_sha256) throw new Error('内联源码 SHA-256 与 source-map 不一致');
  const payload = {
    manifest,
    caseText,
    caseSha256,
    verification,
    sourceMap: map,
    sourceText: source,
  };
  tracePrelude = `  <script>window.__ATLAS_TRACE_BUNDLE__ = {manifest:${inlineScriptJson(manifest)},caseText:${inlineScriptJson(caseText)},caseSha256:${inlineScriptJson(caseSha256)},verification:${inlineScriptJson(verification)}};window.__ATLAS_SOURCE_BUNDLE__ = {map:${inlineScriptJson(map)},source:${inlineScriptJson(source)}};</script>\n`;
  console.log(`inlined trace ${(caseText.length / 1024).toFixed(0)} KB + source ${(source.length / 1024).toFixed(0)} KB, case sha256 ${caseSha256.slice(0, 16)}…`);
  return tracePrelude;
}

let llamaPrelude = null;
function getLlamaPrelude() {
  if (llamaPrelude != null) return llamaPrelude;
  const dataDir = path.join(root, 'data', 'llama');
  const sourceDir = path.join(root, 'sources', 'llama');
  const manifest = JSON.parse(readFileSync(path.join(dataDir, 'manifest.json'), 'utf8'));
  const caseText = readFileSync(path.join(dataDir, 'smollm2-main-002.json'), 'utf8');
  const caseData = JSON.parse(caseText);
  const caseSha256 = createHash('sha256').update(caseText, 'utf8').digest('hex');
  const instance = (manifest.instances || []).find((item) => item.case_id === 'smollm2-main-002');
  if (!instance) throw new Error('清单缺少 smollm2-main-002');
  if (instance.case_sha256 !== caseSha256) throw new Error('内联 LLaMA 案例 SHA-256 与清单不一致');
  if (caseData.schema !== 'atlas.trace.case/2' || caseData.case_id !== 'smollm2-main-002') {
    throw new Error('内联 LLaMA 案例格式/ID 不符合数据契约');
  }
  const verification = JSON.parse(readFileSync(path.join(dataDir, 'verification.json'), 'utf8'));
  const map = JSON.parse(readFileSync(path.join(sourceDir, 'source-map.json'), 'utf8'));
  const texts = {};
  for (const file of Object.keys(map.sources || {})) {
    texts[file] = readFileSync(path.join(sourceDir, file), 'utf8');
    const sha = createHash('sha256').update(texts[file], 'utf8').digest('hex');
    if (sha !== map.sources[file].sha256) throw new Error(`内联源码 ${file} SHA-256 与 source-map 不一致`);
  }
  llamaPrelude = `  <script>window.__ATLAS_LLAMA_BUNDLE__ = {manifest:${inlineScriptJson(manifest)},caseText:${inlineScriptJson(caseText)},caseSha256:${inlineScriptJson(caseSha256)},verification:${inlineScriptJson(verification)}};window.__ATLAS_LLAMA_SOURCE_BUNDLE__ = {map:${inlineScriptJson(map)},texts:${inlineScriptJson(texts)}};</script>\n`;
  console.log(`inlined LLaMA trace ${(caseText.length / 1024).toFixed(0)} KB + sources ${Object.values(texts).reduce((n, t) => n + t.length, 0) / 1024 | 0} KB, case sha256 ${caseSha256.slice(0, 16)}…`);
  return llamaPrelude;
}

const out = await esbuild.build({
  entryPoints: [path.join(root, 'js/main.js')],
  bundle: true,
  write: false,
  format: 'iife', // 普通脚本：file:// 下没有 module 加载与 CORS 问题
  target: ['es2020'],
  minify: true,
  legalComments: 'inline',
  logLevel: 'info',
  plugins: [{
    // 把模板里 importmap 的两条映射交给 esbuild 解析
    name: 'three-alias',
    setup(build) {
      build.onResolve({ filter: /^three$/ }, () => ({ path: path.join(vendor, 'three.module.js') }));
      build.onResolve({ filter: /^three\/addons\// }, (args) => ({
        path: path.join(vendor, 'addons', args.path.replace(/^three\/addons\//, '')),
      }));
    },
  }],
});

let js = out.outputFiles[0].text;
js = js.replace(/<\/script/gi, '<\\/script');

let css = readFileSync(path.join(root, 'css/style.css'), 'utf8');
css = css.replace(/<\/style/gi, '<\\/style');

const MODULE_TAG = '  <script type="module" src="./js/main.js"></script>';

for (const entry of ENTRIES) {
  let html = readFileSync(path.join(root, entry.template), 'utf8');
  const source = entry.template;
  if (!html.includes('window.ATLAS_ENTRY')) {
    throw new Error(`${source} 缺少 window.ATLAS_ENTRY 入口配置，模板可能变了`);
  }

  // 1. 去掉 importmap（打包后不再需要）
  html = html.replace(/  <script type="importmap">[\s\S]*?<\/script>\n/, '');

  // 2. 内联样式
  const styleLink = /  <link rel="stylesheet" href="\.\/css\/style\.css" \/>/;
  if (!styleLink.test(html)) throw new Error(`${source} 的样式表链接没找到，模板可能变了`);
  html = html.replace(styleLink, () => `  <style>\n${css}\n  </style>`);

  // 3. 用打包后的普通脚本替换 module 入口；入口配置的普通 script 原样保留
  if (!html.includes(MODULE_TAG)) throw new Error(`${source} 的入口脚本没找到，模板可能变了`);
  const prelude = entry.inlineData ? getTracePrelude() : (entry.inlineLlamaData ? getLlamaPrelude() : '');
  html = html.replace(MODULE_TAG, () => `${prelude}  <script>\n${js}\n  </script>`);

  // 4. 标题标记为单文件版
  if (!/<title>[^<]*<\/title>/.test(html)) throw new Error(`${source} 的标题没找到`);
  html = html.replace(/<title>[^<]*<\/title>/, () => `<title>${entry.title}（单文件版）</title>`);

  if (html.includes('type="module"') || html.includes('type="importmap"') || html.includes('css/style.css')) {
    throw new Error(`${source} 仍含外部/模块引用，单文件构建未完成`);
  }
  if (entry.inlineData) {
    if (!html.includes('window.__ATLAS_TRACE_BUNDLE__ = {manifest:') || !html.includes('window.__ATLAS_SOURCE_BUNDLE__ = {map:')) {
      throw new Error(`${source} 的内联轨迹/源码缺失`);
    }
  } else if (entry.inlineLlamaData) {
    if (!html.includes('window.__ATLAS_LLAMA_BUNDLE__ = {manifest:') || !html.includes('window.__ATLAS_LLAMA_SOURCE_BUNDLE__ = {map:')) {
      throw new Error(`${source} 的内联 LLaMA 轨迹/源码缺失`);
    }
    if (html.includes('window.__ATLAS_TRACE_BUNDLE__ = {manifest:')) throw new Error(`${source} 不应内联 GPT-2 轨迹`);
  } else if (html.includes('window.__ATLAS_TRACE_BUNDLE__ = {')) {
    throw new Error(`${source} 不应内联 GPT-2 轨迹`);
  }

  const target = path.join(root, entry.output);
  writeFileSync(target, html);
  console.log(`wrote ${target} (${(Buffer.byteLength(html) / 1024).toFixed(0)} KB)`);
}
