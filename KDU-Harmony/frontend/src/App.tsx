import "./styles.css";

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

function App() {
  return (
    <main className="app-shell">
      <section className="workspace">
        <div className="masthead">
          <p className="eyebrow">Healthcare Semantic Search</p>
          <h1>Secure medical record retrieval workspace</h1>
          <p className="summary">
            This scaffold is ready for the next feature slices: authentication, ingestion, OCR,
            PHI-aware processing, hybrid retrieval, and audit logging.
          </p>
        </div>

        <div className="status-grid" aria-label="Platform scaffold status">
          <article>
            <span>Backend</span>
            <strong>{apiBaseUrl}</strong>
          </article>
          <article>
            <span>Search stack</span>
            <strong>OpenSearch planned</strong>
          </article>
          <article>
            <span>Data policy</span>
            <strong>Synthetic records only</strong>
          </article>
        </div>
      </section>
    </main>
  );
}

export default App;
