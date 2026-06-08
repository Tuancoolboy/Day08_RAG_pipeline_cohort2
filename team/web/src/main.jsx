import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  AlertCircle,
  Bot,
  CheckCircle2,
  Database,
  FileText,
  Loader2,
  Search,
  Send,
  Settings2,
  ShieldCheck,
} from 'lucide-react';
import './styles.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const DEFAULT_SAMPLE_QUESTIONS = [
  'Điều 249 quy định gì về tội tàng trữ trái phép chất ma túy?',
  'Các hình thức cai nghiện ma túy theo Luật Phòng, chống ma túy 2021 là gì?',
  'Điều 251 quy định khung hình phạt cơ bản cho mua bán trái phép chất ma túy thế nào?',
  'Khi đọc tin nghệ sĩ liên quan đến ma túy cần kiểm chứng nguồn tin ra sao?',
];

const FALLBACK_MODES = ['hybrid_vector', 'vector', 'hybrid', 'dense_only', 'lexical_only'];

function scoreLabel(score) {
  return Number.isFinite(score) ? score.toFixed(2) : '0.00';
}

function citationLabel(source) {
  const metadata = source?.metadata || {};
  const sourceName = metadata.source || metadata.title || 'Unknown source';
  const year = metadata.year || 'n.d.';
  return `[${sourceName}, ${year}]`;
}

async function requestJson(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
  });

  if (!response.ok) {
    throw new Error(`API error ${response.status}`);
  }

  return response.json();
}

function SourceCard({ source, index }) {
  const metadata = source.metadata || {};
  return (
    <article className="source-card">
      <div className="source-head">
        <div>
          <p className="source-kicker">Source {index + 1}</p>
          <h3>{metadata.title || 'Nguồn chưa đặt tên'}</h3>
        </div>
        <span className="score-pill">Score {scoreLabel(source.score)}</span>
      </div>
      <p className="source-meta">
        {metadata.source || 'Unknown source'} · {metadata.year || 'n.d.'} ·{' '}
        {metadata.doc_type || 'document'}
      </p>
      <p className="source-content">{source.content}</p>
    </article>
  );
}

function App() {
  const [question, setQuestion] = useState(DEFAULT_SAMPLE_QUESTIONS[0]);
  const [topK, setTopK] = useState(5);
  const [retrievalMode, setRetrievalMode] = useState('hybrid_vector');
  const [useReranking, setUseReranking] = useState(true);
  const [useOpenai, setUseOpenai] = useState(true);
  const [health, setHealth] = useState(null);
  const [vectorStore, setVectorStore] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const retrievalModes = health?.retrieval_modes?.length ? health.retrieval_modes : FALLBACK_MODES;
  const sampleQuestions = health?.sample_questions?.length
    ? health.sample_questions
    : DEFAULT_SAMPLE_QUESTIONS;

  const citations = useMemo(() => {
    const seen = new Set();
    const items = [];
    for (const source of result?.sources || []) {
      const label = citationLabel(source);
      if (!seen.has(label)) {
        seen.add(label);
        items.push(label);
      }
    }
    return items;
  }, [result]);

  useEffect(() => {
    async function loadStatus() {
      try {
        const [healthData, vectorData] = await Promise.all([
          requestJson('/api/health'),
          requestJson('/api/vector-store'),
        ]);
        setHealth(healthData);
        setVectorStore(vectorData);
      } catch (apiError) {
        setError(
          `Không kết nối được API tại ${API_BASE_URL}. Kiểm tra backend rồi tải lại trang.`
        );
      }
    }

    loadStatus();
  }, []);

  async function submitQuestion(event) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) {
      setError('Vui lòng nhập câu hỏi.');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const data = await requestJson('/api/chat', {
        method: 'POST',
        body: JSON.stringify({
          query: trimmed,
          top_k: topK,
          retrieval_mode: retrievalMode,
          use_reranking: useReranking,
          use_openai: useOpenai,
        }),
      });
      setResult(data);
      setHistory((items) => [{ question: trimmed, result: data }, ...items].slice(0, 6));
    } catch (apiError) {
      setError(`Không tạo được câu trả lời: ${apiError.message}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="hero">
        <div className="hero-copy">
          <div className="eyebrow">
            <ShieldCheck size={16} aria-hidden="true" />
            Day08 RAG Pipeline
          </div>
          <h1>RAG Chatbot pháp luật ma túy</h1>
          <p>
            Tra cứu quy định, đối chiếu nguồn và xem mức độ liên quan của từng tài liệu
            trong cùng một màn hình làm việc.
          </p>
        </div>
        <div className="status-grid" aria-label="System status">
          <div className="status-item">
            <Database size={20} aria-hidden="true" />
            <div>
              <span>Vector DB</span>
              <strong>{vectorStore?.record_count ?? 0} records</strong>
            </div>
          </div>
          <div className="status-item">
            <Bot size={20} aria-hidden="true" />
            <div>
              <span>Generation</span>
              <strong>{result?.metadata?.mode || 'ready'}</strong>
            </div>
          </div>
          <div className="status-item">
            <CheckCircle2 size={20} aria-hidden="true" />
            <div>
              <span>Backend</span>
              <strong>{health?.status || 'checking'}</strong>
            </div>
          </div>
        </div>
      </section>

      {error ? (
        <div className="alert" role="alert">
          <AlertCircle size={18} aria-hidden="true" />
          {error}
        </div>
      ) : null}

      <section className="workspace">
        <div className="query-panel">
          <div className="panel-title">
            <Search size={19} aria-hidden="true" />
            <h2>Đặt câu hỏi</h2>
          </div>

          <form onSubmit={submitQuestion}>
            <label htmlFor="question">Câu hỏi</label>
            <textarea
              id="question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Nhập câu hỏi về pháp luật ma túy..."
              rows={6}
            />

            <div className="controls">
              <label>
                Retrieval mode
                <select value={retrievalMode} onChange={(event) => setRetrievalMode(event.target.value)}>
                  {retrievalModes.map((mode) => (
                    <option key={mode} value={mode}>
                      {mode}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                Top K
                <input
                  type="number"
                  min="1"
                  max="8"
                  value={topK}
                  onChange={(event) => setTopK(Number(event.target.value))}
                />
              </label>
            </div>

            <div className="toggles" aria-label="Pipeline switches">
              <label className="switch-row">
                <input
                  type="checkbox"
                  checked={useReranking}
                  onChange={(event) => setUseReranking(event.target.checked)}
                />
                Reranking
              </label>
              <label className="switch-row">
                <input
                  type="checkbox"
                  checked={useOpenai}
                  onChange={(event) => setUseOpenai(event.target.checked)}
                />
                Dùng OpenAI nếu có key
              </label>
            </div>

            <button className="primary-button" type="submit" disabled={loading}>
              {loading ? <Loader2 className="spin" size={18} aria-hidden="true" /> : <Send size={18} aria-hidden="true" />}
              {loading ? 'Đang xử lý' : 'Gửi câu hỏi'}
            </button>
          </form>

          <div className="samples">
            <h3>Câu hỏi mẫu</h3>
            <div className="sample-list">
              {sampleQuestions.map((sample) => (
                <button key={sample} type="button" onClick={() => setQuestion(sample)}>
                  {sample}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="answer-panel">
          <div className="panel-title">
            <FileText size={19} aria-hidden="true" />
            <h2>Câu trả lời</h2>
          </div>

          {result ? (
            <>
              <div className="answer-card">
                <p>{result.answer}</p>
                <div className="answer-meta">
                  <span>{result.metadata?.mode || 'unknown'}</span>
                  <span>{result.metadata?.retrieval_mode || retrievalMode}</span>
                </div>
              </div>

              <section className="citation-section" aria-label="Citations">
                <h3>Citations</h3>
                <div className="citation-list">
                  {citations.map((citation) => (
                    <span key={citation}>{citation}</span>
                  ))}
                </div>
              </section>
            </>
          ) : (
            <div className="empty-state">
              <Bot size={26} aria-hidden="true" />
              <p>Câu trả lời sẽ xuất hiện tại đây.</p>
            </div>
          )}
        </div>
      </section>

      <section className="sources-section">
        <div className="panel-title">
          <Settings2 size={19} aria-hidden="true" />
          <h2>Source documents</h2>
        </div>
        <div className="source-grid">
          {(result?.sources || []).length ? (
            result.sources.map((source, index) => (
              <SourceCard key={`${source.metadata?.id || index}-${index}`} source={source} index={index} />
            ))
          ) : (
            <div className="empty-state source-empty">
              <FileText size={24} aria-hidden="true" />
              <p>Source documents sẽ xuất hiện tại đây.</p>
            </div>
          )}
        </div>
      </section>

      {history.length ? (
        <section className="history-section">
          <h2>Lịch sử gần đây</h2>
          {history.map((item) => (
            <button
              className="history-item"
              key={item.question}
              type="button"
              onClick={() => {
                setQuestion(item.question);
                setResult(item.result);
              }}
            >
              {item.question}
            </button>
          ))}
        </section>
      ) : null}
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
