import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "@/components/Sidebar";

export const metadata: Metadata = {
  title: "AI Scribe — Provider Portal",
  description: "Medical documentation powered by AI",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased flex min-h-screen">
        <Sidebar />
        <main className="flex-1 overflow-auto bg-[#F8FAFC]">
          {children}
        </main>
      </body>
    </html>
  );
}
