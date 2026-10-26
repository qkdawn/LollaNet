import type { Metadata } from 'next';
import './globals.css';
import 'katex/dist/katex.min.css';
export const metadata: Metadata = { title: 'LollaNet · 模型与曲线', description: '跨学科模型与 Behavior Mode 工作台' };
export default function Layout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="zh-CN"><body>{children}</body></html>;
}
