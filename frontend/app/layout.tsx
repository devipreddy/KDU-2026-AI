import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Voyager ChatKit Lab",
  description: "Server-driven travel booking chat powered by ChatKit and FastAPI.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
