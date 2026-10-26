import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default function Markdown({ text }: { text: string }) {
  return <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[[rehypeKatex, { trust: false, strict: 'warn', throwOnError: false }]]} components={{
    a: ({ children, href }) => <a href={href} {...(href?.startsWith('#') ? {} : { target: '_blank', rel: 'noreferrer' })}>{children}</a>,
  }}>{text}</ReactMarkdown>;
}
