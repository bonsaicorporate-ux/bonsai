import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

type Msg = { role: "user" | "assistant"; text: string };

export default function App() {
  const [ready, setReady] = useState(false);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const r = await invoke<string>("init_model");
        setReady(r === "ready");
      } catch (e) {
        console.error("init_model failed:", e);
      }
    })();

    const un = listen<string>("token", (e) => {
      setMessages((m) => {
        const last = m[m.length - 1];
        if (!last || last.role !== "assistant") return [...m, { role: "assistant", text: e.payload }];
        const copy = m.slice(0, -1);
        return [...copy, { ...last, text: last.text + e.payload }];
      });
    });

    return () => { un.then((f) => f()); };
  }, []);

  async function onSend(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || !ready) return;
    const prompt = input.trim();
    setMessages((m) => [...m, { role: "user", text: prompt }, { role: "assistant", text: "" }]);
    setInput("");
    try {
      await invoke("generate", { prompt });
    } catch (err) {
      console.error("generate failed:", err);
    }
  }

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column", fontFamily: "system-ui" }}>
      <header style={{ padding: 12, borderBottom: "1px solid #eee" }}>
        BonsAI {ready ? "— model loaded" : "— loading…"}
      </header>
      <main style={{ flex: 1, overflowY: "auto", padding: 16 }}>
        {messages.map((m, i) => (
          <div key={i} style={{ display: "flex", justifyContent: m.role === "user" ? "flex-end" : "flex-start", margin: "8px 0" }}>
            <div style={{ background: m.role === "user" ? "#1e90ff22" : "#eee", padding: "10px 12px", borderRadius: 10, whiteSpace: "pre-wrap", maxWidth: 720 }}>
              {m.text || "…"}
            </div>
          </div>
        ))}
      </main>
      <form onSubmit={onSend} style={{ display: "flex", gap: 8, padding: 12, borderTop: "1px solid #eee" }}>
        <input value={input} onChange={(e) => setInput(e.target.value)} placeholder={ready ? "Say something…" : "Loading model…"}
               disabled={!ready} style={{ flex: 1, padding: "10px 12px", borderRadius: 8, border: "1px solid #ddd" }} />
        <button type="submit" disabled={!ready} style={{ padding: "10px 14px", borderRadius: 8, border: "1px solid #1e90ff", background: "#1e90ff", color: "#fff" }}>
          Send
        </button>
      </form>
    </div>
  );
} 