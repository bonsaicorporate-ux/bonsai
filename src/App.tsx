import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";

type Msg = { role: "user" | "assistant"; text: string };

export default function App() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");

  async function onSend(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim()) return;

    const userText = input.trim();
    setMessages((m) => [...m, { role: "user", text: userText }]);
    setInput("");

    // call our Rust command
    const reply = await invoke<string>("echo", { text: userText });
    setMessages((m) => [...m, { role: "assistant", text: reply }]);
  }

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column", fontFamily: "system-ui, -apple-system" }}>
      <header style={{ padding: "12px 16px", borderBottom: "1px solid #eee", fontWeight: 600 }}>
        BonsAI (echo mode)
      </header>

      <main style={{ flex: 1, overflowY: "auto", padding: 16 }}>
        {messages.length === 0 && (
          <div style={{ color: "#666" }}>Type below—assistant will echo your message for now.</div>
        )}
        {messages.map((m, i) => (
          <div key={i} style={{ margin: "8px 0", display: "flex", justifyContent: m.role === "user" ? "flex-end" : "flex-start" }}>
            <div
              style={{
                maxWidth: 640,
                padding: "10px 12px",
                borderRadius: 12,
                background: m.role === "user" ? "#1e90ff22" : "#eee",
                whiteSpace: "pre-wrap",
              }}
            >
              {m.text}
            </div>
          </div>
        ))}
      </main>

      <form onSubmit={onSend} style={{ display: "flex", gap: 8, padding: 12, borderTop: "1px solid #eee" }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Say something…"
          autoFocus
          style={{ flex: 1, padding: "10px 12px", borderRadius: 10, border: "1px solid #ddd" }}
        />
        <button type="submit" style={{ padding: "10px 14px", borderRadius: 10, border: "1px solid #1e90ff", background: "#1e90ff", color: "white" }}>
          Send
        </button>
      </form>
    </div>
  );
}