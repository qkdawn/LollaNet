import { test, expect } from '@playwright/test';
import { experiments } from '../lib/dynamics';

test('eight dynamics topics render interactive plots and safe equations', async ({ page }, info) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/?view=curves');
  await page.getByRole('tab', { name: /动力学实验/ }).click();
  await expect(page.locator('.experiment-card')).toHaveCount(8);
  await page.screenshot({ path: `test-results/${info.project.name}-dynamics-directory.png`, fullPage: true });
  for (const experiment of experiments) {
    await test.step(experiment.name, async () => {
      await page.goto(`/?view=curves&atlas=dynamics&experiment=${experiment.id}`);
      await expect(page.locator('.experiment-intro h2')).toHaveText(experiment.question);
      await expect(page.locator('.experiment-result')).toBeVisible();
      await expect(page.locator('.trajectory').first()).toHaveAttribute('d', /^M.+L/);
      // Relay thresholds change the output, while the prescribed input stays fixed.
      const first = page.locator('.trajectory').nth(experiment.id === 'hysteresis' ? 1 : 0), before = await first.getAttribute('d');
      const parameter = experiment.parameters.find(p => p.key !== 'initialOn')!;
      await page.getByRole('spinbutton', { name: parameter.label, exact: true }).fill(String(parameter.value === parameter.max ? parameter.min : parameter.max));
      await expect(page).toHaveURL(/settings=/);
      await expect(first).not.toHaveAttribute('d', before!);
      await page.getByRole('button', { name: '重置实验', exact: true }).click();
      await expect(first).toHaveAttribute('d', before!);
      await page.locator('.experiment-equations summary').click();
      await expect(page.locator('.experiment-equations .katex')).toBeVisible();
      await expect(page.locator('.katex-error')).toHaveCount(0);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
      for (const path of await page.locator('.trajectory').evaluateAll(nodes => nodes.map(n => n.getAttribute('d')))) expect(path).not.toMatch(/NaN|Infinity/);
      if (experiment.id === 'multistability' || experiment.id === 'resonance') {
        await page.evaluate(() => scrollTo(0,0));
        await page.screenshot({ path: `test-results/${info.project.name}-dynamics-${experiment.id}.png`, fullPage: true });
      }
    });
  }
  expect(errors).toEqual([]);
});

test('presets, disturbance, URL restoration, graph selection and cross-navigation', async ({ page }) => {
  await page.goto('/?view=curves&atlas=dynamics&experiment=stability');
  await page.getByRole('button', { name: '山顶 · 远离', exact: true }).click();
  await expect(page.locator('.experiment-result')).toContainText('不稳定');
  await page.reload();
  await expect(page.getByRole('spinbutton', { name: '地形曲率', exact: true })).toHaveValue('-1');
  await page.goBack();
  await expect(page.getByRole('spinbutton', { name: '地形曲率', exact: true })).toHaveValue('1');
  const before = await page.locator('.trajectory').first().getAttribute('d');
  await page.getByRole('button', { name: /施加一次速度扰动/ }).click();
  await expect(page).toHaveURL(/kick=1/);
  await expect(page.locator('.trajectory').first()).not.toHaveAttribute('d', before!);
  await page.reload();
  await expect(page.locator('.experiment-result')).toContainText('速度增加 1');
  await page.getByRole('button', { name: '撤销扰动', exact: true }).click();
  await expect(page.locator('.trajectory').first()).toHaveAttribute('d', before!);

  await page.goto('/?view=curves&atlas=dynamics&experiment=bifurcation');
  await page.locator('.dynamics-panel').filter({ has: page.getByRole('heading', { name: '长期半径的分岔图' }) }).locator('svg').click({ position: { x: 70, y: 70 } });
  await expect(page.getByRole('spinbutton', { name: '控制参数 μ', exact: true })).not.toHaveValue('0.3');
  await page.goto('/?view=curves&atlas=dynamics&experiment=resonance');
  await page.locator('.dynamics-panel').filter({ has: page.getByRole('heading', { name: '理论稳态频率响应' }) }).locator('svg').click({ position: { x: 70, y: 70 } });
  await expect(page.getByRole('spinbutton', { name: '驱动角频率 Ω', exact: true })).not.toHaveValue('1');
  await page.goto('/?view=curves&atlas=dynamics&experiment=multistability');
  await page.getByRole('img', { name: /初始位置与速度的吸引域/ }).click({ position: { x: 70, y: 70 } });
  await expect(page.getByRole('spinbutton', { name: '初始位置', exact: true })).not.toHaveValue('1');
  await page.reload();
  await expect(page.locator('.experiment-result')).toBeVisible();

  await page.goto('/?view=curves&mode=oscillation&variant=oscillation-damped');
  await page.locator('.curve-experiments').getByRole('button', { name: '受迫响应与共振', exact: true }).click();
  await expect(page).toHaveURL(/experiment=resonance/);
  await page.goBack();
  await expect(page.locator('.plot-panel h2')).toHaveText('衰减振荡');
  await page.getByRole('tab', { name: /动力学实验/ }).click();
  await page.getByRole('textbox', { name: '搜索', exact: true }).fill('延迟');
  await page.getByRole('textbox', { name: '搜索', exact: true }).press('Enter');
  await expect(page.locator('.experiment-card')).toHaveCount(1);
  await page.locator('.experiment-card').click();
  await expect(page.locator('.experiment-intro h2')).toContainText('过度纠正');
  await page.goto('/?view=curves&atlas=dynamics&experiment=unknown');
  await expect(page.getByRole('heading', { name: '动力学实验不存在' })).toBeVisible();
  await page.getByRole('button', { name: '返回动力学实验' }).click();
  await expect(page.locator('.experiment-card')).toHaveCount(8);
});

test('invalid URL parameters recover and mobile drawer exposes the topics', async ({ page }, info) => {
  await page.goto('/?view=curves&atlas=dynamics&experiment=stability&settings=null');
  await expect(page.getByRole('spinbutton', { name: '地形曲率', exact: true })).toHaveValue('1');
  if (info.project.name === 'mobile') {
    await page.getByRole('button', { name: '分类导航', exact: true }).click();
    await page.locator('.sidebar.open').getByRole('button', { name: '耦合与同步', exact: true }).click();
    await expect(page.locator('.sidebar.open')).toHaveCount(0);
    await expect(page.locator('.experiment-intro h2')).toContainText('锁定相位差');
  }
  await page.getByRole('navigation').getByRole('button', { name: '模型库', exact: true }).click();
  await expect(page).not.toHaveURL(/atlas=|experiment=|settings=/);
  await expect(page.locator('.model-row')).toHaveCount(88);
});
