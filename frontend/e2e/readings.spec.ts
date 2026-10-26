import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import type { Library } from '../lib/types';

const library: Library = JSON.parse(readFileSync(new URL('../../backend/data/library.json', import.meta.url), 'utf8'));
const batches: { modelIds: string[] }[] = JSON.parse(readFileSync(new URL('../content/reading-batches.json', import.meta.url), 'utf8'));
const readings = batches.flatMap(batch => batch.modelIds).map(id => {
  const model = library.models.find(model => model.id === id);
  if (!model?.reading) throw new Error(`Missing published reading: ${id}`);
  return model;
});

test('all readings open as long-form articles with original material and rendered math', async ({ page }, info) => {
  test.setTimeout(240_000);
  const errors: string[] = [];
  page.on('pageerror', err => errors.push(err.message));
  for (const model of readings) {
    const reading = model.reading!;
    await test.step(model.name, async () => {
      await page.goto(`/?view=models&model=${model.id}`);
      await expect(page.locator('.reading-header h2')).toHaveText(reading.title);
      if (model.status === 'candidate' || model.status === 'contested') {
        await expect(page.locator(`.reading-header .badge.${model.status}`)).toHaveText(model.status === 'candidate' ? '候选' : '有争议');
      }
      await expect(page.getByRole('tab', { name: '新版讲解 · 试读' })).toHaveAttribute('aria-selected', 'true');
      await expect(page.locator('.model-results')).toBeHidden();
      const opening = reading.body.split('\n\n').find(paragraph => !paragraph.startsWith('#'))!;
      await expect(page.locator('.reading-body').first()).toContainText(opening.slice(0, 35));
      const width = await page.locator('.model-detail').evaluate(el => el.getBoundingClientRect().width);
      expect(width).toBeLessThanOrEqual(721);
      await expect(page.locator('.reading-appendix')).not.toHaveAttribute('open');
      await page.locator('.reading-appendix summary').click();
      await expect(page.locator('.reading-appendix .katex').first()).toBeVisible();
      await expect(page.locator('.katex-error')).toHaveCount(0);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
      await page.getByRole('tab', { name: '原始资料', exact: true }).click();
      await expect(page.locator('#original-panel h2')).toHaveText(model.name);
      await expect(page.locator('#original-panel')).toContainText('核心机制');
      await page.getByRole('tab', { name: '新版讲解 · 试读' }).click();
      await expect(page.locator('.reading-header h2')).toHaveText(reading.title);
    });
  }
  await page.evaluate(() => scrollTo(0, 0));
  await page.screenshot({ path: `test-results/${info.project.name}-reading.png`, fullPage: true });
  expect(errors).toEqual([]);
});

test('summary search, citation navigation, safe external links and fallback', async ({ page }, info) => {
  await page.goto('/?view=models&q=一千人的账');
  await expect(page.locator('.model-row')).toHaveCount(1);
  await expect(page.locator('.model-row')).toContainText('新版讲解');
  await page.locator('.model-row').click();
  const citation = page.locator('.reading-body a[href="#source-bayes"]');
  await expect(citation).not.toHaveAttribute('target');
  await citation.click();
  await expect(page).toHaveURL(/#source-bayes$/);
  await expect(page.locator('#source-bayes')).toBeInViewport();
  await expect(page.locator('#source-bayes > a')).toHaveAttribute('href', 'https://plato.stanford.edu/entries/bayes-theorem/');
  await expect(page.locator('#source-bayes > a')).toHaveAttribute('rel', /noreferrer/);
  await page.locator('.reading-appendix summary').click();
  await page.evaluate(() => scrollTo(0, 0));
  await page.screenshot({ path: `test-results/${info.project.name}-base-rate-reading.png`, fullPage: true });
  await page.getByRole('tab', { name: '原始资料', exact: true }).click();
  await expect(page.locator('#original-panel .katex').first()).toBeVisible();
  await page.getByRole('button', { name: '返回列表', exact: true }).click();
  await expect(page.locator('.model-row')).toHaveCount(1);
  // Keep fallback coverage when every current model has a reading.
  await page.route('**/api/models/mm-bayesian-updating', async route => {
    const response = await route.fetch();
    const model = await response.json();
    delete model.reading;
    await route.fulfill({ response, json: model });
  });
  await page.goto('/?view=models&model=mm-bayesian-updating');
  await expect(page.locator('.reading-tabs')).toHaveCount(0);
  await expect(page.locator('.model-detail')).toContainText('核心机制');
});
