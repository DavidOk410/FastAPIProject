console.log("JS loaded");

document.addEventListener("DOMContentLoaded", () => {
  const chatEl = document.getElementById("chat");
  const qEl = document.getElementById("q");
  const sendBtn = document.getElementById("send");
  const clearBtn = document.getElementById("clear");
  const apiUrlEl = document.getElementById("apiUrl");
  const docIdEl = document.getElementById("docId");

  let history = [];

  function addMsg(role, content, extraMeta) {
    const div = document.createElement("div");
    div.className = `msg ${role === "user" ? "user" : "assistant"}`;
    div.textContent = `${role === "user" ? "You" : "Assistant"}: ${content}`;

    if (extraMeta) {
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = extraMeta;
      div.appendChild(meta);
    }

    chatEl.appendChild(div);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  function setBusy(busy) {
    sendBtn.disabled = busy;
    clearBtn.disabled = busy;
    qEl.disabled = busy;
  }

  async function loadDocs() {
  try {
    const res = await fetch("/docs-list");
    const docs = await res.json();

    docs.forEach(d => {
      const opt = document.createElement("option");
      opt.value = d.doc_id;
      opt.textContent = d.doc_id;
      docIdEl.appendChild(opt);
    });

  } catch (e) {
    console.error("Failed to load docs list", e);
  }
}

  loadDocs();
  async function send() {
    const question = qEl.value.trim();
    if (!question) return;

    addMsg("user", question);
    history.push({ role: "user", content: question });
    qEl.value = "";
    setBusy(true);

    const payload = {
      question,
      history,
    };

    const doc_id = docIdEl.value.trim();
    if (doc_id) payload.doc_id = doc_id;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => {controller.abort("Request timed out after 90 seconds");}, 90000);


      const res = await fetch(apiUrlEl.value.trim(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      const text = await res.text();
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${text}`);
      }

      let data;
      try {
        data = JSON.parse(text);
      } catch {
        throw new Error(`Server returned non-JSON: ${text}`);
      }

      const answer = data.answer ?? "(no answer)";
      const conf = typeof data.confidence === "number" ? data.confidence.toFixed(2) : "n/a";

      addMsg("assistant", answer, `confidence=${conf} sources=${(data.sources || []).length}`);

      history = Array.isArray(data.history) ? data.history : history;
      history.push({ role: "assistant", content: answer });
    } catch (e) {
        let msg = "Unknown error";

        if (e instanceof Error) {
          msg = e.message;
        } else if (typeof e === "string") {
          msg = e;
        } else if (e && typeof e === "object") {
          msg = JSON.stringify(e);
        }

        if (e?.name === "AbortError") {
          msg = "The request took too long and was cancelled.";
        }

        addMsg("assistant", "ERROR calling API: " + msg);
        console.error(e);
      } finally {
      setBusy(false);
      qEl.focus();
    }
  }

  function clearChat() {
    history = [];
    chatEl.innerHTML = "";
    qEl.value = "";
    qEl.focus();
  }

  sendBtn.addEventListener("click", (e) => {
    e.preventDefault();
    send();
  });

  qEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      send();
    }
  });

  clearBtn.addEventListener("click", (e) => {
    e.preventDefault();
    clearChat();
  });
});