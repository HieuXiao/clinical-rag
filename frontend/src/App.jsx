import { useState, useRef, useEffect, useCallback } from 'react';
import './App.css';

const rawApiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8001';
const API_BASE_URL = rawApiBaseUrl.replace('0.0.0.0', '127.0.0.1');

const FALLBACK_MODELS = [
  'gemini-2.5-flash',
  'gemini-2.5-flash-lite',
  'gemini-1.5-flash',
  'gemini-1.5-pro',
];

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [modelStats, setModelStats] = useState([]);
  const [selectedModel, setSelectedModel] = useState(FALLBACK_MODELS[0]);
  const [statusMessage, setStatusMessage] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const loadModels = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/models`);
      if (!response.ok) throw new Error('Không tải được model list');
      const data = await response.json();
      if (Array.isArray(data.models) && data.models.length > 0) {
        setModelStats(data.models);
        setSelectedModel((prev) => (data.models.some((m) => m.name === prev) ? prev : data.models[0].name));
        return;
      }
    } catch (error) {
      console.warn(error);
    }

    setModelStats(
      FALLBACK_MODELS.map((name) => ({
        name,
        used: 0,
        limit: 20,
        usage_percent: 0,
      }))
    );
  }, []);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  const isRateLimitError = (message) => {
    const text = (message || '').toLowerCase();
    return text.includes('rate limit') || text.includes('429') || text.includes('quota');
  };

  const getNextModel = (currentModel) => {
    const source = modelStats.length > 0 ? modelStats.map((m) => m.name) : FALLBACK_MODELS;
    const index = source.indexOf(currentModel);
    if (index < 0 || index === source.length - 1) return null;
    return source[index + 1];
  };

  const sendChat = async (modelName) => {
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: input,
        model: modelName,
        api_key: apiKey,
      }),
    });

    if (!response.ok) {
      let detail = 'Lỗi máy chủ';
      try {
        const err = await response.json();
        detail = err.detail || detail;
      } catch (e) {
        console.warn(e);
      }
      throw new Error(detail);
    }

    return response.json();
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      setStatusMessage(`Đang dùng model ${selectedModel}...`);
      let usedModel = selectedModel;
      let data;

      try {
        data = await sendChat(usedModel);
      } catch (firstError) {
        if (!isRateLimitError(firstError.message)) throw firstError;

        const fallback = getNextModel(usedModel);
        if (!fallback) throw firstError;

        setStatusMessage(`Model ${usedModel} hết quota, chuyển sang ${fallback}...`);
        usedModel = fallback;
        setSelectedModel(fallback);
        data = await sendChat(fallback);
      }

      setMessages((prev) => [
        ...prev,
        { 
          role: 'bot', 
          content: data.answer || 'Không có phản hồi',
          sources: data.sources || []
        }
      ]);
      setStatusMessage(`Đã trả lời bằng ${usedModel}`);
      loadModels();
    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        { role: 'bot', content: `Lỗi: ${error.message || 'Không kết nối được tới máy chủ.'}` }
      ]);
      setStatusMessage('Có lỗi khi gửi câu hỏi');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🏥 Trợ lý Y Tế RAG - SEAL Hackathon</h1>
        <div className="settings-container">
          <div className="api-key-container">
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Nhập API Key của bạn"
              className="api-key-input"
            />
          </div>
          <div className="model-selector-container">
            <select 
              value={selectedModel} 
              onChange={(e) => setSelectedModel(e.target.value)}
              className="model-selector"
            >
              {(modelStats.length > 0 ? modelStats : FALLBACK_MODELS.map((name) => ({ name, usage_percent: 0 }))).map((model) => (
                <option key={model.name || model} value={model.name || model}>
                  {(model.name || model)} - {model.usage_percent ?? 0}%
                </option>
              ))}
            </select>
          </div>
        </div>
        {statusMessage && <div className="status-bar">{statusMessage}</div>}
      </header>
      
      <main className="chat-container">
        {messages.length === 0 ? (
          <div className="empty-state">
            <p>Bắt đầu trò chuyện bằng cách nhập câu hỏi của bạn xuống bên dưới.</p>
          </div>
        ) : (
          <div className="message-list">
            {messages.map((msg, idx) => (
              <div key={idx} className={`message ${msg.role}`}>
                <div className="message-content">
                  <p>{msg.content}</p>
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="sources">
                      <strong>Nguồn:</strong>
                      <ul>
                        {msg.sources.map((src, i) => (
                          <li key={i}>{src}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {loading && (
              <div className="message bot">
                <div className="message-content loading">
                   Đang suy nghĩ...
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </main>

      <footer className="app-footer">
        <form onSubmit={handleSubmit} className="input-form">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Nhập câu hỏi tra cứu..."
            disabled={loading}
          />
          <button type="submit" disabled={loading || !input.trim()}>
            Gửi
          </button>
        </form>
      </footer>
    </div>
  );
}

export default App;
