"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Props {
  content: string;
  className?: string;
}

export default function MarkdownViewer({ content, className = "" }: Props) {
  return (
    <div
      className={`prose prose-sm max-w-none prose-headings:text-gray-800 prose-table:text-sm ${className}`}
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
    </div>
  );
}
