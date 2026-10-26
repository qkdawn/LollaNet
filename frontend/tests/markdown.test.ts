import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import Markdown from '../components/markdown';

test('math renders without accepting embedded HTML or trusted TeX commands', () => {
  const safe = renderToStaticMarkup(createElement(Markdown, { text: '$$\nP=\\frac{9}{108}\n$$' }));
  assert(safe.includes('class="katex"')); assert(!safe.includes('katex-error'));
  const unsafe = renderToStaticMarkup(createElement(Markdown, { text: '<script>alert(1)</script>\n\n[bad](javascript:alert(1))\n\n$\\href{javascript:alert(1)}{bad}$' }));
  assert(!unsafe.includes('<script>')); assert(!unsafe.includes('href="javascript:'));
});
