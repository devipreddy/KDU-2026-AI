import {
  ChangeEvent,
  type Dispatch,
  FormEvent,
  type SetStateAction,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import "./styles.css";

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const demoPassword = "ChangeMe123!";
const defaultMinimumConfidence = 0;

type ViewId = "search" | "ingestion" | "audit";
type RoleId = "doctor" | "nurse" | "admin" | "researcher" | "records_staff";
type ConfidenceLevel = "high" | "medium" | "low";

type RoleProfile = {
  label: string;
  email: string;
  displayMode: string;
  visibility: "full" | "limited" | "de_identified" | "metadata_only" | "operational";
};

type RenderedChunk = {
  chunk_id: string;
  section: string | null;
  page_number: number | null;
  start_offset: number | null;
  end_offset: number | null;
  token_count: number | null;
  chunk_type: string | null;
  text: string | null;
};

type RenderedCitation = {
  document_id: string;
  external_id: string;
  source_document: string;
  source_uri: string | null;
  document_type: string;
  page_number: number | null;
  section: string | null;
  hospital: string | null;
  physician: string | null;
  visit_id: string | null;
  checksum_sha256: string;
  citation_label: string;
  diagnosis: string | null;
  icd_codes: string[];
  visit_date: string | null;
};

type RetrievalConfidence = {
  score: number;
  level: ConfidenceLevel;
  reranker_score: number | null;
  hybrid_score: number;
  ocr_confidence: number | null;
  source_count: number;
};

type SearchHit = {
  final_rank: number;
  patient_display_ref: string | null;
  matched_chunk: RenderedChunk;
  parent_context: RenderedChunk | null;
  citation: RenderedCitation;
  confidence: RetrievalConfidence;
  retrieval: Record<string, unknown>;
  redactions: string[];
};

type SearchAnswer = {
  status: string;
  answer: string;
  provider: string | null;
  model: string | null;
  citations: Array<Record<string, unknown>>;
  latency_ms: number | null;
  error: string | null;
};

type PipelineMetadata = {
  authorization?: {
    role_names?: string[];
    denied?: boolean;
    deny_reason?: string | null;
    allowed_sensitivity_levels?: string[];
  };
  bm25?: { hit_count?: number; candidate_count?: number };
  dense?: {
    hit_count?: number;
    candidate_count?: number;
    embedding_model?: string;
    embedding_dimension?: number;
  };
  fusion?: { algorithm?: string; hit_count?: number; rrf_k?: number };
  reranker?: { model?: string; reranked_count?: number; rerank_top_n?: number };
  context_expansion?: { hit_count?: number; strategy?: string };
  rendering?: { render_mode?: string; phi_visibility?: string };
  llm?: { status?: string; provider?: string | null; model?: string | null; latency_ms?: number };
};

type SearchResponse = {
  query: string;
  hit_count: number;
  answer: SearchAnswer;
  hits: SearchHit[];
  timeline: Array<Record<string, unknown>>;
  pipeline: PipelineMetadata;
};

type SseMessage = {
  event: string;
  data: string;
};

type IngestionJob = {
  id: string;
  status: string;
  stage: string;
  attempts: number;
  error_message: string | null;
  queued_at: string | null;
  started_at: string | null;
  finished_at: string | null;
};

type DocumentChunkPreview = {
  id: string;
  parent_chunk_id: string | null;
  chunk_index: number;
  section: string | null;
  chunk_type: string | null;
  indexing_status: string;
  embedding_id: string | null;
  token_count: number | null;
  ocr_confidence: number | null;
  content_preview: string;
};

type IngestionDocument = {
  id: string;
  external_id: string;
  patient_ref: string;
  visit_id: string | null;
  document_type: string;
  status: string;
  file_name: string;
  source_uri: string;
  mime_type: string;
  checksum_sha256: string;
  hospital: string | null;
  physician: string | null;
  department: string | null;
  diagnosis: string | null;
  icd_codes: string[];
  sensitivity_level: string;
  is_encrypted: boolean;
  ocr_required: boolean;
  ocr_engine: string | null;
  ocr_confidence: number | null;
  extraction_status: string;
  review_status: string;
  chunk_count: number;
  indexed_chunk_count: number;
  latest_job: IngestionJob | null;
  created_at: string;
  updated_at: string;
};

type IngestionDocumentDetail = IngestionDocument & {
  extracted_text: string | null;
  extracted_text_char_count: number;
  metadata: Record<string, unknown>;
  jobs: IngestionJob[];
  chunks: DocumentChunkPreview[];
};

type UploadCandidate = {
  file: File;
  relativePath: string;
  patientRef: string;
  visitId: string;
  documentType: string;
  hospital: string;
  physician: string;
  department: string;
  metadata: Record<string, string>;
};

type AuditEvent = {
  id: string;
  occurred_at: string;
  actor_email: string | null;
  actor_display_name: string | null;
  role_snapshot: string[];
  action: string;
  query_text: string | null;
  resource_type: string | null;
  resource_id: string | null;
  result_document_ids: string[];
  decision: string | null;
  ip_address: string | null;
  user_agent: string | null;
  metadata: Record<string, unknown>;
};

const roleProfiles: Record<RoleId, RoleProfile> = {
  doctor: {
    label: "Doctor",
    email: "doctor@example.com",
    displayMode: "Full PHI when assigned",
    visibility: "full",
  },
  nurse: {
    label: "Nurse",
    email: "nurse@example.com",
    displayMode: "Limited clinical when assigned",
    visibility: "limited",
  },
  admin: {
    label: "Admin",
    email: "admin@example.com",
    displayMode: "Metadata only",
    visibility: "metadata_only",
  },
  researcher: {
    label: "Researcher",
    email: "researcher@example.com",
    displayMode: "De-identified",
    visibility: "de_identified",
  },
  records_staff: {
    label: "Records",
    email: "records@example.com",
    displayMode: "Operational full PHI",
    visibility: "operational",
  },
};

const documentKindLabels: Record<string, string> = {
  clinical_note: "Clinical note",
  discharge_summary: "Discharge summary",
  handwritten_note: "Handwritten note",
  lab_report: "Lab report",
  prescription: "Prescription",
  scanned_pdf: "Scanned PDF",
  typed_pdf: "Typed PDF",
};

const auditActionLabels: Record<string, string> = {
  access_denied: "Access denied",
  break_glass: "Break glass",
  document_read: "Document read",
  document_upload: "Document upload",
  login: "Login",
  phi_decrypt: "PHI decrypt",
  query_run: "Query run",
};

const auditDateFormatter = new Intl.DateTimeFormat("en-US", {
  dateStyle: "medium",
  timeStyle: "short",
});

function App() {
  const tokenCache = useRef<Partial<Record<RoleId, string>>>({});
  const [activeView, setActiveView] = useState<ViewId>("search");
  const [query, setQuery] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [role, setRole] = useState<RoleId>("doctor");
  const [documentKind, setDocumentKind] = useState("all");
  const [hospital, setHospital] = useState("all");
  const [minimumConfidence, setMinimumConfidence] = useState(defaultMinimumConfidence);
  const [searchResponse, setSearchResponse] = useState<SearchResponse | null>(null);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [auditEvents, setAuditEvents] = useState<AuditEvent[]>([]);
  const [auditSearch, setAuditSearch] = useState("");
  const [auditRole, setAuditRole] = useState("all");
  const [auditAction, setAuditAction] = useState("all");
  const [auditDecision, setAuditDecision] = useState("all");
  const [auditError, setAuditError] = useState<string | null>(null);
  const [isLoadingAudit, setIsLoadingAudit] = useState(false);
  const [uploadCandidates, setUploadCandidates] = useState<UploadCandidate[]>([]);
  const [ingestionDocuments, setIngestionDocuments] = useState<IngestionDocument[]>([]);
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null);
  const [selectedDocument, setSelectedDocument] = useState<IngestionDocumentDetail | null>(null);
  const [ingestionError, setIngestionError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isLoadingDocuments, setIsLoadingDocuments] = useState(false);
  const [isLoadingDocumentDetail, setIsLoadingDocumentDetail] = useState(false);
  const [activeDocumentAction, setActiveDocumentAction] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState("");

  const activeRole = roleProfiles[role];

  const ensureToken = useCallback(async (selectedRole: RoleId, forceRefresh = false) => {
    const cachedToken = tokenCache.current[selectedRole];
    if (cachedToken && !forceRefresh) {
      return cachedToken;
    }

    const response = await fetch(`${apiBaseUrl}/api/v1/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        email: roleProfiles[selectedRole].email,
        password: demoPassword,
      }),
    });
    if (!response.ok) {
      throw new Error(await responseErrorMessage(response, "Login failed"));
    }

    const body = (await response.json()) as { access_token: string };
    tokenCache.current[selectedRole] = body.access_token;
    return body.access_token;
  }, []);

  const authenticatedFetch = useCallback(
    async (selectedRole: RoleId, input: RequestInfo | URL, init: RequestInit = {}) => {
      let token = await ensureToken(selectedRole);
      let response = await fetch(input, withBearerToken(init, token));
      if (response.status !== 401) {
        return response;
      }

      delete tokenCache.current[selectedRole];
      token = await ensureToken(selectedRole, true);
      response = await fetch(input, withBearerToken(init, token));
      return response;
    },
    [ensureToken],
  );

  const executeSearch = useCallback(
    async (nextQuery: string, selectedRole: RoleId) => {
      const trimmedQuery = nextQuery.trim();
      if (!trimmedQuery) {
        setSearchError("Enter a search query.");
        return;
      }

      setIsSearching(true);
      setSearchError(null);
      setSearchResponse(null);
      setSubmittedQuery(trimmedQuery);
      setHasSearched(true);
      try {
        const response = await authenticatedFetch(
          selectedRole,
          `${apiBaseUrl}/api/v1/search/stream`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              query: trimmedQuery,
              limit: 5,
              candidate_limit: 20,
              rerank_top_n: 8,
              include_llm_answer: true,
            }),
          },
        );
        if (!response.ok) {
          throw new Error(await responseErrorMessage(response, "Search failed"));
        }
        if (!response.body) {
          throw new Error("Streaming search response was empty.");
        }
        await readSearchStream(response, setSearchResponse);
      } catch (error) {
        setSearchError(error instanceof Error ? error.message : "Search failed");
      } finally {
        setIsSearching(false);
      }
    },
    [authenticatedFetch],
  );

  const fetchAuditEvents = useCallback(async () => {
    setIsLoadingAudit(true);
    setAuditError(null);
    try {
      const response = await authenticatedFetch(
        "admin",
        `${apiBaseUrl}/api/v1/audit/events?limit=200`,
      );
      if (!response.ok) {
        throw new Error(await responseErrorMessage(response, "Audit fetch failed"));
      }
      const body = (await response.json()) as AuditEvent[];
      setAuditEvents(body);
    } catch (error) {
      setAuditError(error instanceof Error ? error.message : "Audit fetch failed");
    } finally {
      setIsLoadingAudit(false);
    }
  }, [authenticatedFetch]);

  const fetchDocuments = useCallback(async () => {
    setIsLoadingDocuments(true);
    setIngestionError(null);
    try {
      const response = await authenticatedFetch(
        "records_staff",
        `${apiBaseUrl}/api/v1/documents?source=ingestion_dashboard&limit=250`,
      );
      if (!response.ok) {
        throw new Error(await responseErrorMessage(response, "Document fetch failed"));
      }
      const body = (await response.json()) as IngestionDocument[];
      setIngestionDocuments(body);
      if (body.length === 0) {
        setSelectedDocumentId(null);
        setSelectedDocument(null);
      }
    } catch (error) {
      setIngestionError(error instanceof Error ? error.message : "Document fetch failed");
    } finally {
      setIsLoadingDocuments(false);
    }
  }, [authenticatedFetch]);

  const fetchDocumentDetail = useCallback(
    async (documentId: string) => {
      setIngestionError(null);
      setIsLoadingDocumentDetail(true);
      try {
        const response = await authenticatedFetch(
          "records_staff",
          `${apiBaseUrl}/api/v1/documents/${documentId}`,
        );
        if (!response.ok) {
          throw new Error(await responseErrorMessage(response, "Document detail fetch failed"));
        }
        const body = (await response.json()) as IngestionDocumentDetail;
        setSelectedDocumentId(documentId);
        setSelectedDocument(body);
      } catch (error) {
        setIngestionError(error instanceof Error ? error.message : "Document detail fetch failed");
      } finally {
        setIsLoadingDocumentDetail(false);
      }
    },
    [authenticatedFetch],
  );

  async function runDocumentAction(
    documentId: string,
    action: "extract" | "approve",
  ): Promise<IngestionDocumentDetail> {
    const response = await authenticatedFetch(
      "records_staff",
      `${apiBaseUrl}/api/v1/documents/${documentId}/${action}`,
      { method: "POST" },
    );
    if (!response.ok) {
      throw new Error(await responseErrorMessage(response, `${action} failed`));
    }
    return (await response.json()) as IngestionDocumentDetail;
  }

  async function handleDocumentAction(documentId: string, action: "extract" | "approve") {
    setActiveDocumentAction(`${action}:${documentId}`);
    setIngestionError(null);
    try {
      const detail = await runDocumentAction(documentId, action);
      setSelectedDocumentId(documentId);
      setSelectedDocument(detail);
      await fetchDocuments();
    } catch (error) {
      setIngestionError(error instanceof Error ? error.message : `${action} failed`);
    } finally {
      setActiveDocumentAction(null);
    }
  }

  async function runBulkAction(action: "extract" | "approve") {
    const documents =
      action === "extract"
        ? ingestionDocuments.filter((document) => document.extraction_status !== "extracted")
        : ingestionDocuments.filter((document) => document.review_status === "pending");
    setActiveDocumentAction(`bulk:${action}`);
    setIngestionError(null);
    try {
      for (const [index, document] of documents.entries()) {
        setUploadProgress(`${action} ${index + 1}/${documents.length}: ${document.file_name}`);
        const detail = await runDocumentAction(document.id, action);
        setSelectedDocumentId(document.id);
        setSelectedDocument(detail);
      }
      setUploadProgress("");
      await fetchDocuments();
    } catch (error) {
      setIngestionError(error instanceof Error ? error.message : `${action} failed`);
    } finally {
      setActiveDocumentAction(null);
      setUploadProgress("");
    }
  }

  useEffect(() => {
    if (activeView === "ingestion") {
      void fetchDocuments();
    }
  }, [activeView, fetchDocuments]);

  useEffect(() => {
    if (activeView === "audit") {
      void fetchAuditEvents();
    }
  }, [activeView, fetchAuditEvents]);

  useEffect(() => {
    if (activeView !== "ingestion" || ingestionDocuments.length === 0) {
      return;
    }
    if (
      selectedDocumentId &&
      ingestionDocuments.some((document) => document.id === selectedDocumentId)
    ) {
      return;
    }
    void fetchDocumentDetail(ingestionDocuments[0].id);
  }, [activeView, fetchDocumentDetail, ingestionDocuments, selectedDocumentId]);

  const filteredResults = useMemo(
    () =>
      (searchResponse?.hits ?? []).filter((hit) => {
        if (documentKind !== "all" && hit.citation.document_type !== documentKind) {
          return false;
        }
        if (hospital !== "all" && hit.citation.hospital !== hospital) {
          return false;
        }
        return hit.confidence.score * 100 >= minimumConfidence;
      }),
    [documentKind, hospital, minimumConfidence, searchResponse],
  );

  const hospitals = useMemo(
    () =>
      Array.from(
        new Set((searchResponse?.hits ?? []).map((hit) => hit.citation.hospital).filter(Boolean)),
      ).sort() as string[],
    [searchResponse],
  );

  const documentKinds = useMemo(
    () =>
      Array.from(new Set((searchResponse?.hits ?? []).map((hit) => hit.citation.document_type)))
        .filter(Boolean)
        .sort(),
    [searchResponse],
  );

  const filteredAuditEvents = useMemo(
    () =>
      auditEvents.filter((event) => {
        if (auditRole !== "all" && !event.role_snapshot.includes(auditRole)) {
          return false;
        }
        if (auditAction !== "all" && event.action !== auditAction) {
          return false;
        }
        if (auditDecision !== "all" && event.decision !== auditDecision) {
          return false;
        }
        return matchesAuditSearch(event, auditSearch);
      }),
    [auditAction, auditDecision, auditEvents, auditRole, auditSearch],
  );

  const auditSummary = useMemo(() => buildAuditSummary(filteredAuditEvents), [filteredAuditEvents]);
  const ingestionSummary = useMemo(
    () => buildIngestionSummary(ingestionDocuments),
    [ingestionDocuments],
  );
  const pendingExtractionCount = ingestionDocuments.filter(
    (document) => document.extraction_status !== "extracted",
  ).length;
  const pendingApprovalCount = ingestionDocuments.filter(
    (document) => document.review_status === "pending",
  ).length;
  const topResult = filteredResults[0];
  const retrievedResultCount = searchResponse?.hits.length ?? 0;
  const filtersHideRetrievedResults =
    !isSearching && retrievedResultCount > 0 && filteredResults.length === 0;
  const isAppBusy =
    isSearching ||
    isLoadingAudit ||
    isUploading ||
    isLoadingDocuments ||
    isLoadingDocumentDetail ||
    activeDocumentAction !== null;

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void executeSearch(query, role);
  }

  function handleRoleChange(nextRole: RoleId) {
    setRole(nextRole);
    setSearchResponse(null);
    setSearchError(null);
    setSubmittedQuery("");
    setHasSearched(false);
    resetFilters();
  }

  function resetFilters() {
    setDocumentKind("all");
    setHospital("all");
    setMinimumConfidence(defaultMinimumConfidence);
  }

  function resetAuditFilters() {
    setAuditSearch("");
    setAuditRole("all");
    setAuditAction("all");
    setAuditDecision("all");
  }

  function handleDatasetFiles(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []).filter((file) =>
      file.name.toLowerCase().endsWith(".pdf"),
    );
    setUploadCandidates(
      uniqueUploadCandidates(files).map((file, index) => buildUploadCandidate(file, index)),
    );
    setUploadProgress("");
    setIngestionError(null);
  }

  async function uploadDatasetFiles() {
    setIsUploading(true);
    setIngestionError(null);
    try {
      for (const [index, candidate] of uploadCandidates.entries()) {
        setUploadProgress(`upload ${index + 1}/${uploadCandidates.length}: ${candidate.file.name}`);
        const formData = new FormData();
        formData.append("file", candidate.file);
        formData.append("patient_ref", candidate.patientRef);
        formData.append("visit_id", candidate.visitId);
        formData.append("document_type", candidate.documentType);
        formData.append("hospital", candidate.hospital);
        formData.append("physician", candidate.physician);
        formData.append("department", candidate.department);
        formData.append("sensitivity_level", "high");
        formData.append("metadata", JSON.stringify(candidate.metadata));

        const response = await authenticatedFetch(
          "records_staff",
          `${apiBaseUrl}/api/v1/documents/upload`,
          {
            method: "POST",
            body: formData,
          },
        );
        if (!response.ok) {
          throw new Error(await responseErrorMessage(response, "Upload failed"));
        }
        setUploadProgress(
          `uploaded ${index + 1}/${uploadCandidates.length}: ${candidate.file.name}`,
        );
      }
      setUploadCandidates([]);
      setUploadProgress("");
      await fetchDocuments();
    } catch (error) {
      setIngestionError(error instanceof Error ? error.message : "Upload failed");
    } finally {
      setIsUploading(false);
      setUploadProgress("");
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Healthcare Semantic Search</p>
          <h1>Medical record retrieval</h1>
        </div>
        <nav className="view-tabs" aria-label="Workspace views">
          <button
            className={activeView === "search" ? "selected" : ""}
            onClick={() => setActiveView("search")}
            type="button"
          >
            Search
          </button>
          <button
            className={activeView === "ingestion" ? "selected" : ""}
            onClick={() => setActiveView("ingestion")}
            type="button"
          >
            Ingestion
          </button>
          <button
            className={activeView === "audit" ? "selected" : ""}
            onClick={() => setActiveView("audit")}
            type="button"
          >
            Audit log
          </button>
        </nav>
        <div className="connection-status">
          <span>API</span>
          <strong>{apiBaseUrl}</strong>
        </div>
      </header>
      {isAppBusy ? <BusyBar /> : null}

      {activeView === "search" ? (
        <section className="search-workspace" aria-label="Search workspace">
          <aside className="filter-panel" aria-label="Search filters">
            <div className="panel-heading">
              <span>Role</span>
              <strong>{activeRole.displayMode}</strong>
            </div>

            <div className="role-segment expanded" aria-label="Role selector">
              {(Object.keys(roleProfiles) as RoleId[]).map((roleId) => (
                <button
                  className={role === roleId ? "selected" : ""}
                  key={roleId}
                  onClick={() => handleRoleChange(roleId)}
                  type="button"
                >
                  {roleProfiles[roleId].label}
                </button>
              ))}
            </div>

            <div className="signed-in">
              <span>Signed in</span>
              <strong>{activeRole.email}</strong>
            </div>

            <label className="field">
              <span>Document type</span>
              <select
                value={documentKind}
                onChange={(event) => setDocumentKind(event.target.value)}
              >
                <option value="all">All types</option>
                {documentKinds.map((value) => (
                  <option key={value} value={value}>
                    {formatDocumentKind(value)}
                  </option>
                ))}
              </select>
            </label>

            <label className="field">
              <span>Hospital</span>
              <select value={hospital} onChange={(event) => setHospital(event.target.value)}>
                <option value="all">All hospitals</option>
                {hospitals.map((hospitalName) => (
                  <option key={hospitalName} value={hospitalName}>
                    {hospitalName}
                  </option>
                ))}
              </select>
            </label>

            <label className="field">
              <span>Minimum confidence</span>
              <input
                max="95"
                min="0"
                onChange={(event) => setMinimumConfidence(Number(event.target.value))}
                type="range"
                value={minimumConfidence}
              />
              <output>{minimumConfidence}%</output>
            </label>

            <button className="secondary-action" onClick={resetFilters} type="button">
              Reset filters
            </button>
          </aside>

          <section className="results-area" aria-busy={isSearching} aria-label="Search results">
            <form className="query-panel" onSubmit={handleSubmit}>
              <label htmlFor="query">Natural language query</label>
              <div className="query-row">
                <textarea
                  id="query"
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="patients with cardiac issues treated in Q1 2025"
                  rows={3}
                  value={query}
                />
                <button disabled={isSearching || !query.trim()} type="submit">
                  {isSearching ? <InlineLoader label="Searching" /> : "Search"}
                </button>
              </div>
            </form>

            {searchError ? <div className="status-banner error">{searchError}</div> : null}

            {hasSearched || isSearching ? (
              <PipelineStrip
                pipeline={searchResponse?.pipeline ?? null}
                loading={isSearching && !searchResponse}
              />
            ) : null}

            {isSearching && !searchResponse ? (
              <AnswerSkeleton />
            ) : (
              <AnswerPanel answer={searchResponse?.answer ?? null} />
            )}

            {hasSearched || isSearching ? (
              <div className="result-summary" aria-live="polite">
                <div>
                  <span>
                    {isSearching
                      ? "Pipeline running"
                      : `${filteredResults.length}/${retrievedResultCount} visible`}
                  </span>
                  <strong>{searchResponse?.query ?? submittedQuery}</strong>
                </div>
                <div className={`masking-mode ${activeRole.visibility}`}>
                  {searchAccessLabel(searchResponse, filteredResults, activeRole)}
                </div>
              </div>
            ) : null}

            {hasSearched || isSearching ? (
              <div className="content-grid">
                <section className="result-list" aria-label="Result list">
                  {isSearching ? (
                    <ResultSkeletonList />
                  ) : (
                    filteredResults.map((result) => (
                      <ResultCard key={result.matched_chunk.chunk_id} result={result} />
                    ))
                  )}

                  {!isSearching && filteredResults.length === 0 ? (
                    <div className="empty-state">
                      <strong>
                        {filtersHideRetrievedResults
                          ? "Retrieved records hidden by filters"
                          : "No matching records"}
                      </strong>
                      <span>
                        {filtersHideRetrievedResults
                          ? `${retrievedResultCount} records came back from retrieval, but the current document type, hospital, or confidence filters hide them.`
                          : "Adjust the query, role, or filters."}
                      </span>
                      {filtersHideRetrievedResults ? (
                        <button className="table-action" onClick={resetFilters} type="button">
                          Show retrieved records
                        </button>
                      ) : null}
                    </div>
                  ) : null}
                </section>

                <aside className="detail-panel" aria-label="Selected result context">
                  <div className="panel-heading">
                    <span>Top context</span>
                    <strong>
                      {isSearching
                        ? "Building context"
                        : topResult
                          ? topResult.citation.external_id
                          : "No result"}
                    </strong>
                  </div>
                  {isSearching ? (
                    <ContextSkeleton />
                  ) : topResult ? (
                    <ContextPreview result={topResult} />
                  ) : null}
                </aside>
              </div>
            ) : null}
          </section>
        </section>
      ) : activeView === "ingestion" ? (
        <section className="ingestion-workspace" aria-label="Ingestion workspace">
          <section className="upload-panel" aria-label="Dataset upload">
            <div>
              <span>Dataset intake</span>
              <strong>{uploadCandidates.length || ingestionDocuments.length} documents</strong>
            </div>
            <label className="dataset-picker" htmlFor="dataset-files">
              <span>PDF folder</span>
              <input
                {...{ directory: "", webkitdirectory: "" }}
                accept="application/pdf,.pdf"
                id="dataset-files"
                multiple
                onChange={handleDatasetFiles}
                type="file"
              />
            </label>
            <button
              className="secondary-action compact-action"
              disabled={isUploading || uploadCandidates.length === 0}
              onClick={() => void uploadDatasetFiles()}
              type="button"
            >
              {isUploading ? <InlineLoader label="Uploading" /> : "Upload"}
            </button>
            <button
              className="secondary-action compact-action"
              disabled={isLoadingDocuments}
              onClick={() => void fetchDocuments()}
              type="button"
            >
              {isLoadingDocuments ? <InlineLoader label="Refreshing" /> : "Refresh"}
            </button>
          </section>

          {uploadProgress ? (
            <div className="status-banner info">
              <InlineLoader label={uploadProgress} />
            </div>
          ) : null}
          {ingestionError ? <div className="status-banner error">{ingestionError}</div> : null}

          <section className="audit-metrics" aria-label="Ingestion summary">
            <Metric label="Documents" value={String(ingestionSummary.documents)} />
            <Metric label="Extracted" value={String(ingestionSummary.extracted)} />
            <Metric label="Review" value={String(ingestionSummary.reviewPending)} />
            <Metric label="Indexed" value={String(ingestionSummary.indexed)} />
          </section>

          <section className="ingestion-actions" aria-label="Ingestion actions">
            <button
              className="table-action"
              disabled={activeDocumentAction !== null || pendingExtractionCount === 0}
              onClick={() => void runBulkAction("extract")}
              type="button"
            >
              {activeDocumentAction === "bulk:extract" ? (
                <InlineLoader label="Extracting" />
              ) : (
                "Extract pending"
              )}
            </button>
            <button
              className="table-action"
              disabled={activeDocumentAction !== null || pendingApprovalCount === 0}
              onClick={() => void runBulkAction("approve")}
              type="button"
            >
              {activeDocumentAction === "bulk:approve" ? (
                <InlineLoader label="Indexing" />
              ) : (
                "Approve pending"
              )}
            </button>
          </section>

          <section className="ingestion-layout">
            <section className="document-table-panel" aria-label="Uploaded documents">
              <div className="table-heading">
                <div>
                  <span>Documents</span>
                  <strong>
                    {isLoadingDocuments ? "Loading" : `${ingestionDocuments.length} records`}
                  </strong>
                </div>
              </div>
              <div className="table-scroll">
                <table className="audit-table document-table">
                  <thead>
                    <tr>
                      <th>File</th>
                      <th>Pipeline</th>
                      <th>OCR</th>
                      <th>Chunks</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {isLoadingDocuments && ingestionDocuments.length === 0 ? (
                      <TableSkeletonRows columns={5} rows={5} />
                    ) : (
                      ingestionDocuments.map((document) => (
                        <DocumentRow
                          actionKey={activeDocumentAction}
                          document={document}
                          key={document.id}
                          onApprove={() => void handleDocumentAction(document.id, "approve")}
                          onExtract={() => void handleDocumentAction(document.id, "extract")}
                          onSelect={() => void fetchDocumentDetail(document.id)}
                          selected={selectedDocumentId === document.id}
                        />
                      ))
                    )}
                    {!isLoadingDocuments && ingestionDocuments.length === 0 ? (
                      <tr>
                        <td className="table-empty" colSpan={5}>
                          No uploaded documents.
                        </td>
                      </tr>
                    ) : null}
                  </tbody>
                </table>
              </div>
            </section>

            <aside className="extraction-panel" aria-label="Extraction preview">
              <div className="panel-heading">
                <span>Extraction output</span>
                <strong>{selectedDocument?.external_id ?? "No document"}</strong>
              </div>
              {isLoadingDocumentDetail ? (
                <DocumentPreviewSkeleton />
              ) : selectedDocument ? (
                <DocumentPreview
                  document={selectedDocument}
                  onApprove={() => void handleDocumentAction(selectedDocument.id, "approve")}
                  onExtract={() => void handleDocumentAction(selectedDocument.id, "extract")}
                  working={activeDocumentAction}
                />
              ) : null}
            </aside>
          </section>
        </section>
      ) : (
        <section className="audit-workspace" aria-label="Audit log workspace">
          <section className="audit-controls" aria-label="Audit filters">
            <label className="audit-search-field" htmlFor="audit-search">
              <span>Search audit logs</span>
              <input
                id="audit-search"
                onChange={(event) => setAuditSearch(event.target.value)}
                placeholder="User, document, query, patient ref, IP"
                type="search"
                value={auditSearch}
              />
            </label>

            <label className="field compact">
              <span>Role</span>
              <select value={auditRole} onChange={(event) => setAuditRole(event.target.value)}>
                <option value="all">All roles</option>
                {(Object.keys(roleProfiles) as RoleId[]).map((roleId) => (
                  <option key={roleId} value={roleId}>
                    {roleProfiles[roleId].label}
                  </option>
                ))}
              </select>
            </label>

            <label className="field compact">
              <span>Action</span>
              <select value={auditAction} onChange={(event) => setAuditAction(event.target.value)}>
                <option value="all">All actions</option>
                {Object.entries(auditActionLabels).map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </label>

            <label className="field compact">
              <span>Decision</span>
              <select
                value={auditDecision}
                onChange={(event) => setAuditDecision(event.target.value)}
              >
                <option value="all">All decisions</option>
                <option value="allow">Allow</option>
                <option value="deny">Deny</option>
                <option value="review">Review</option>
              </select>
            </label>

            <button
              className="secondary-action audit-reset"
              disabled={isLoadingAudit}
              onClick={() => void fetchAuditEvents()}
              type="button"
            >
              {isLoadingAudit ? <InlineLoader label="Refreshing" /> : "Refresh"}
            </button>
          </section>

          {auditError ? <div className="status-banner error">{auditError}</div> : null}

          <section className="audit-metrics" aria-label="Audit summary">
            <Metric label="Events" value={String(auditSummary.events)} />
            <Metric label="Documents" value={String(auditSummary.documents)} />
            <Metric label="Denied" value={String(auditSummary.denied)} />
            <Metric label="PHI decrypts" value={String(auditSummary.phiDecrypts)} />
          </section>

          <section className="audit-table-panel" aria-label="Audit event table">
            <div className="table-heading">
              <div>
                <span>Access history</span>
                <strong>
                  {isLoadingAudit ? "Loading" : `${filteredAuditEvents.length} events`}
                </strong>
              </div>
              <button className="table-action" onClick={resetAuditFilters} type="button">
                Reset filters
              </button>
            </div>

            <div className="table-scroll">
              <table className="audit-table">
                <thead>
                  <tr>
                    <th>When</th>
                    <th>Who</th>
                    <th>Action</th>
                    <th>What</th>
                    <th>Query</th>
                    <th>Decision</th>
                    <th>Masking</th>
                  </tr>
                </thead>
                <tbody>
                  {isLoadingAudit && auditEvents.length === 0 ? (
                    <TableSkeletonRows columns={7} rows={6} />
                  ) : (
                    filteredAuditEvents.map((event) => <AuditRow event={event} key={event.id} />)
                  )}

                  {!isLoadingAudit && filteredAuditEvents.length === 0 ? (
                    <tr>
                      <td className="table-empty" colSpan={7}>
                        No audit events match the current filters.
                      </td>
                    </tr>
                  ) : null}
                </tbody>
              </table>
            </div>
          </section>
        </section>
      )}
    </main>
  );
}

async function readSearchStream(
  response: Response,
  setSearchResponse: Dispatch<SetStateAction<SearchResponse | null>>,
) {
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("Streaming search response was empty.");
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let streamedAnswer = "";

  let streamOpen = true;
  while (streamOpen) {
    const { done, value } = await reader.read();
    if (done) {
      streamOpen = false;
      continue;
    }
    buffer += decoder.decode(value, { stream: true });
    const parsed = parseSseBuffer(buffer);
    buffer = parsed.remaining;
    for (const message of parsed.messages) {
      streamedAnswer = applySearchStreamMessage(message, streamedAnswer, setSearchResponse);
    }
  }

  buffer += decoder.decode();
  if (buffer.trim()) {
    const parsed = parseSseBuffer(`${buffer}\n\n`);
    for (const message of parsed.messages) {
      streamedAnswer = applySearchStreamMessage(message, streamedAnswer, setSearchResponse);
    }
  }
}

function parseSseBuffer(buffer: string): { messages: SseMessage[]; remaining: string } {
  const normalized = buffer.replace(/\r\n/g, "\n");
  const blocks = normalized.split("\n\n");
  const remaining = blocks.pop() ?? "";
  const messages = blocks
    .map((block) => parseSseBlock(block))
    .filter((message): message is SseMessage => message !== null);
  return { messages, remaining };
}

function parseSseBlock(block: string): SseMessage | null {
  const lines = block.split("\n");
  let event = "message";
  const dataLines: string[] = [];
  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart());
    }
  }
  if (dataLines.length === 0) {
    return null;
  }
  return { event, data: dataLines.join("\n") };
}

function applySearchStreamMessage(
  message: SseMessage,
  streamedAnswer: string,
  setSearchResponse: Dispatch<SetStateAction<SearchResponse | null>>,
) {
  if (message.event === "retrieval") {
    const nextResponse = JSON.parse(message.data) as SearchResponse;
    setSearchResponse(nextResponse);
    return nextResponse.answer.answer ?? "";
  }

  if (message.event === "answer_delta") {
    const payload = JSON.parse(message.data) as { delta?: string };
    const nextAnswer = streamedAnswer + (payload.delta ?? "");
    setSearchResponse((previous) =>
      previous
        ? {
            ...previous,
            answer: {
              ...previous.answer,
              answer: nextAnswer,
              status: "streaming",
            },
            pipeline: {
              ...previous.pipeline,
              llm: {
                ...(previous.pipeline.llm ?? {}),
                status: "streaming",
              },
            },
          }
        : previous,
    );
    return nextAnswer;
  }

  if (message.event === "answer_done") {
    const finalResponse = JSON.parse(message.data) as SearchResponse;
    setSearchResponse(finalResponse);
    return finalResponse.answer.answer ?? streamedAnswer;
  }

  if (message.event === "error") {
    const payload = JSON.parse(message.data) as { message?: string };
    throw new Error(payload.message ?? "Streaming search failed");
  }

  return streamedAnswer;
}

function BusyBar() {
  return (
    <div className="busy-bar" aria-label="Request in progress" role="status">
      <span />
    </div>
  );
}

function InlineLoader({ label }: { label: string }) {
  return (
    <span className="inline-loader">
      <span aria-hidden="true" />
      {label}
    </span>
  );
}

function SkeletonLine({ width = "100%" }: { width?: string }) {
  return <span className="skeleton-line" style={{ width }} />;
}

function AnswerSkeleton() {
  return (
    <article className="answer-panel answer-skeleton" aria-label="Generating answer">
      <div className="answer-header">
        <div>
          <span>LLM answer</span>
          <strong>Generating grounded answer</strong>
        </div>
        <InlineLoader label="Working" />
      </div>
      <SkeletonLine width="94%" />
      <SkeletonLine width="86%" />
      <SkeletonLine width="58%" />
    </article>
  );
}

function ResultSkeletonList() {
  return (
    <>
      {Array.from({ length: 4 }).map((_, index) => (
        <article className="result-card skeleton-card" key={index}>
          <div className="result-card-header">
            <div>
              <SkeletonLine width="96px" />
              <SkeletonLine width="240px" />
            </div>
            <SkeletonLine width="82px" />
          </div>
          <div className="metadata-strip skeleton-metadata">
            <SkeletonLine />
            <SkeletonLine />
            <SkeletonLine />
            <SkeletonLine />
          </div>
          <SkeletonLine width="98%" />
          <SkeletonLine width="88%" />
          <SkeletonLine width="64%" />
        </article>
      ))}
    </>
  );
}

function ContextSkeleton() {
  return (
    <div className="context-preview skeleton-context" aria-label="Loading context">
      <SkeletonLine width="70%" />
      <SkeletonLine width="82%" />
      <SkeletonLine width="66%" />
      <SkeletonLine width="92%" />
      <SkeletonLine width="74%" />
    </div>
  );
}

function TableSkeletonRows({ columns, rows }: { columns: number; rows: number }) {
  return (
    <>
      {Array.from({ length: rows }).map((_, rowIndex) => (
        <tr className="skeleton-table-row" key={rowIndex}>
          {Array.from({ length: columns }).map((__, columnIndex) => (
            <td key={columnIndex}>
              <SkeletonLine width={columnIndex === 0 ? "78%" : "64%"} />
              <SkeletonLine width={columnIndex === 0 ? "52%" : "42%"} />
            </td>
          ))}
        </tr>
      ))}
    </>
  );
}

function DocumentPreviewSkeleton() {
  return (
    <div className="document-preview" aria-label="Loading document preview">
      <div className="document-preview-skeleton">
        <SkeletonLine width="60%" />
        <SkeletonLine width="80%" />
        <SkeletonLine width="48%" />
      </div>
      <div className="extracted-text skeleton-block">
        <SkeletonLine width="94%" />
        <SkeletonLine width="88%" />
        <SkeletonLine width="91%" />
        <SkeletonLine width="62%" />
      </div>
    </div>
  );
}

function PipelineStrip({
  pipeline,
  loading,
}: {
  pipeline: PipelineMetadata | null;
  loading: boolean;
}) {
  const steps = [
    ["BM25", `${pipeline?.bm25?.hit_count ?? 0} hits`],
    ["Dense", `${pipeline?.dense?.hit_count ?? 0} hits`],
    ["RRF", `${pipeline?.fusion?.hit_count ?? 0} fused`],
    ["Cross-encoder", `${pipeline?.reranker?.reranked_count ?? 0} reranked`],
    ["Context", `${pipeline?.context_expansion?.hit_count ?? 0} expanded`],
    ["LLM", pipeline?.llm?.status ?? "pending"],
  ];

  return (
    <section className="pipeline-strip" aria-label="Retrieval pipeline">
      {steps.map(([label, value], index) => (
        <div className={loading ? "running" : ""} key={label}>
          <span>{label}</span>
          <strong>
            {loading && label !== "LLM" ? (
              <InlineLoader label={index < 3 ? "retrieving" : "running"} />
            ) : (
              value
            )}
          </strong>
        </div>
      ))}
    </section>
  );
}

function AnswerPanel({ answer }: { answer: SearchAnswer | null }) {
  if (!answer) {
    return null;
  }
  const isStreaming = answer.status === "streaming";

  return (
    <article className={`answer-panel ${answer.status}`}>
      <div className="answer-header">
        <div>
          <span>LLM answer</span>
          <strong>{answer.model ?? answer.provider ?? answer.status}</strong>
        </div>
        <span className="answer-status">{answer.status.replace(/_/g, " ")}</span>
      </div>
      <p className={isStreaming ? "answer-text streaming" : "answer-text"}>
        {answer.answer || (isStreaming ? "Waiting for the model to begin..." : "")}
      </p>
      {answer.error ? <span className="answer-error">{answer.error}</span> : null}
      {answer.citations.length > 0 ? (
        <div className="answer-citations">
          {answer.citations.map((citation) => (
            <span key={String(citation.index)}>
              [{String(citation.index)}] {String(citation.citation_label ?? "source")}
            </span>
          ))}
        </div>
      ) : null}
    </article>
  );
}

function ResultCard({ result }: { result: SearchHit }) {
  const citation = result.citation;
  const isMetadataOnly = !result.matched_chunk.text;
  const retrievalSources = retrievalSourceLabels(result);
  const renderingLabel = hitRenderingLabel(result);

  return (
    <article className="result-card">
      <div className="result-card-header">
        <div>
          <span className="result-type">{formatDocumentKind(citation.document_type)}</span>
          <h2>{citation.diagnosis ?? citation.external_id}</h2>
        </div>
        <ConfidenceIndicator level={result.confidence.level} score={result.confidence.score} />
      </div>

      <dl className="metadata-strip">
        <div>
          <dt>Patient</dt>
          <dd>{result.patient_display_ref ?? "Hidden"}</dd>
        </div>
        <div>
          <dt>Visit</dt>
          <dd>{citation.visit_date ?? "Unknown"}</dd>
        </div>
        <div>
          <dt>Section</dt>
          <dd>{citation.section ?? result.matched_chunk.section ?? "Unknown"}</dd>
        </div>
        <div>
          <dt>ICD</dt>
          <dd>{citation.icd_codes.length ? citation.icd_codes.join(", ") : "Unknown"}</dd>
        </div>
      </dl>

      {isMetadataOnly ? (
        <div className="metadata-only-view">
          <span>{citation.hospital ?? "Unknown hospital"}</span>
          <span>{citation.physician ?? "Unknown physician"}</span>
          <span>{citation.external_id}</span>
        </div>
      ) : (
        <p className="snippet">{result.matched_chunk.text}</p>
      )}

      <div className="terms-row">
        {renderingLabel ? <span className="render-chip">{renderingLabel}</span> : null}
        {result.redactions.map((redaction) => (
          <span key={redaction}>{redaction.replace(/_/g, " ")}</span>
        ))}
        {result.confidence.ocr_confidence ? (
          <span>OCR {Math.round(result.confidence.ocr_confidence * 100)}%</span>
        ) : null}
      </div>

      <footer className="citation-row">
        <span>{citation.citation_label}</span>
        <span>{retrievalSources.join(" + ")}</span>
      </footer>
    </article>
  );
}

function ContextPreview({ result }: { result: SearchHit }) {
  const citation = result.citation;

  return (
    <div className="context-preview">
      <dl>
        <div>
          <dt>Source</dt>
          <dd>{citation.source_document}</dd>
        </div>
        <div>
          <dt>Hospital</dt>
          <dd>{citation.hospital ?? "Unknown"}</dd>
        </div>
        <div>
          <dt>Physician</dt>
          <dd>{citation.physician ?? "Unknown"}</dd>
        </div>
        <div>
          <dt>Document</dt>
          <dd>{citation.external_id}</dd>
        </div>
      </dl>

      {result.parent_context?.text ? (
        <p>{result.parent_context.text}</p>
      ) : (
        <p className="metadata-note">Clinical text hidden for this role.</p>
      )}
    </div>
  );
}

function AuditRow({ event }: { event: AuditEvent }) {
  return (
    <tr>
      <td>
        <strong>{auditDateFormatter.format(new Date(event.occurred_at))}</strong>
        <span>{event.id}</span>
      </td>
      <td>
        <strong>{event.actor_display_name ?? "System"}</strong>
        <span>{event.actor_email ?? "unknown"}</span>
        <span>{event.role_snapshot.join(", ") || "unknown"}</span>
      </td>
      <td>
        <span className="action-chip">{auditActionLabels[event.action] ?? event.action}</span>
      </td>
      <td>
        <strong>{event.resource_id ?? event.resource_type ?? "retrieval"}</strong>
        <span>{event.resource_type ?? "unknown"}</span>
        <span>{event.result_document_ids.join(", ") || "none"}</span>
      </td>
      <td className="query-cell">
        <strong>{event.query_text ?? "No query"}</strong>
        <span>{metadataString(event, "phi_visibility")}</span>
        <span>{metadataFilterSummary(event)}</span>
      </td>
      <td>
        <span className={`decision-pill ${event.decision ?? "review"}`}>
          {event.decision ?? "review"}
        </span>
        <span>{event.ip_address ?? "unknown"}</span>
      </td>
      <td>
        <strong>{metadataMaskingSummary(event)}</strong>
        <span>{metadataAssignmentSummary(event)}</span>
        <span>{event.user_agent ?? "unknown"}</span>
      </td>
    </tr>
  );
}

function DocumentRow({
  document,
  selected,
  actionKey,
  onSelect,
  onExtract,
  onApprove,
}: {
  document: IngestionDocument;
  selected: boolean;
  actionKey: string | null;
  onSelect: () => void;
  onExtract: () => void;
  onApprove: () => void;
}) {
  const extractionRunning = actionKey === `extract:${document.id}`;
  const approvalRunning = actionKey === `approve:${document.id}`;
  const canExtract = document.extraction_status !== "extracted" && !extractionRunning;
  const canApprove = document.review_status === "pending" && !approvalRunning;
  const extractLabel = extractionRunning
    ? "Extracting"
    : document.extraction_status === "extracted"
      ? "Extracted"
      : "Extract";

  return (
    <tr className={selected ? "selected-row" : ""}>
      <td>
        <button className="row-link" onClick={onSelect} type="button">
          {document.file_name}
        </button>
        <span>{document.patient_ref}</span>
        <span>{document.hospital ?? "unknown hospital"}</span>
      </td>
      <td>
        <strong>{document.status}</strong>
        <span>{document.extraction_status}</span>
        <span>{document.review_status}</span>
      </td>
      <td>
        <strong>{formatDocumentKind(document.document_type)}</strong>
        <span>
          {document.ocr_required ? (document.ocr_engine ?? "OCR required") : "typed text"}
        </span>
        <span>{formatPercent(document.ocr_confidence)}</span>
      </td>
      <td>
        <strong>
          {document.indexed_chunk_count}/{document.chunk_count}
        </strong>
        <span>{document.latest_job?.stage ?? "no job"}</span>
        {document.latest_job?.error_message ? (
          <span className="error-inline">{document.latest_job.error_message}</span>
        ) : null}
      </td>
      <td>
        <div className="row-actions">
          <button disabled={!canExtract || actionKey !== null} onClick={onExtract} type="button">
            {extractionRunning ? <InlineLoader label={extractLabel} /> : extractLabel}
          </button>
          <button disabled={!canApprove || actionKey !== null} onClick={onApprove} type="button">
            {approvalRunning ? <InlineLoader label="Approving" /> : "Approve"}
          </button>
        </div>
      </td>
    </tr>
  );
}

function DocumentPreview({
  document,
  working,
  onExtract,
  onApprove,
}: {
  document: IngestionDocumentDetail;
  working: string | null;
  onExtract: () => void;
  onApprove: () => void;
}) {
  const canExtract = document.extraction_status !== "extracted";
  const canApprove = document.review_status === "pending";
  const extractLabel =
    working === `extract:${document.id}`
      ? "Extracting"
      : document.extraction_status === "extracted"
        ? "Extracted"
        : "Extract text";

  return (
    <div className="document-preview">
      <dl>
        <div>
          <dt>Status</dt>
          <dd>{document.status}</dd>
        </div>
        <div>
          <dt>Type</dt>
          <dd>{formatDocumentKind(document.document_type)}</dd>
        </div>
        <div>
          <dt>Hospital</dt>
          <dd>{document.hospital ?? "Unknown"}</dd>
        </div>
        <div>
          <dt>Physician</dt>
          <dd>{document.physician ?? "Unknown"}</dd>
        </div>
      </dl>

      <div className="preview-actions">
        <button disabled={!canExtract || working !== null} onClick={onExtract} type="button">
          {working === `extract:${document.id}` ? (
            <InlineLoader label={extractLabel} />
          ) : (
            extractLabel
          )}
        </button>
        <button disabled={!canApprove || working !== null} onClick={onApprove} type="button">
          {working === `approve:${document.id}` ? (
            <InlineLoader label="Indexing" />
          ) : (
            "Approve and index"
          )}
        </button>
      </div>

      <div className="preview-meta">
        <span>{document.extracted_text_char_count} chars</span>
        <span>{document.chunk_count} chunks</span>
        <span>{document.indexed_chunk_count} indexed</span>
      </div>

      {document.latest_job?.error_message ? (
        <div className="status-banner error">{document.latest_job.error_message}</div>
      ) : null}

      <pre className="extracted-text">
        {document.extracted_text ?? "No extraction output for this document yet."}
      </pre>

      <div className="chunk-preview-list">
        {document.chunks.slice(0, 8).map((chunk) => (
          <article key={chunk.id}>
            <strong>
              {chunk.section ?? "Document"} / {chunk.chunk_type ?? "chunk"}
            </strong>
            <span>{chunk.indexing_status}</span>
            <p>{chunk.content_preview}</p>
          </article>
        ))}
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ConfidenceIndicator({ level, score }: { level: ConfidenceLevel; score: number }) {
  return (
    <div className={`confidence ${level}`} aria-label={`Confidence ${Math.round(score * 100)}%`}>
      <span>{Math.round(score * 100)}%</span>
      <div>
        <i style={{ width: `${score * 100}%` }} />
      </div>
      <strong>{level}</strong>
    </div>
  );
}

async function responseErrorMessage(response: Response, fallback: string) {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") {
      return body.detail;
    }
    if (body.detail) {
      return JSON.stringify(body.detail);
    }
  } catch {
    return fallback;
  }
  return fallback;
}

function withBearerToken(init: RequestInit, token: string): RequestInit {
  const headers = new Headers(init.headers);
  headers.set("Authorization", `Bearer ${token}`);
  return {
    ...init,
    headers,
  };
}

function formatDocumentKind(value: string) {
  return documentKindLabels[value] ?? value.replace(/_/g, " ");
}

function retrievalSourceLabels(result: SearchHit) {
  const rawSources = result.retrieval.sources;
  const sources = Array.isArray(rawSources) ? rawSources.map((item) => String(item)) : [];
  const labels = sources.map((source) => {
    if (source === "bm25") {
      return "BM25";
    }
    if (source === "dense") {
      return "Dense";
    }
    return source;
  });
  if (result.confidence.reranker_score !== null) {
    labels.push("Cross-encoder");
  }
  return labels.length ? labels : ["Retrieved"];
}

function hitRenderingLabel(result: SearchHit) {
  const rendering = result.retrieval.rendering;
  if (!rendering || typeof rendering !== "object") {
    return null;
  }
  const renderingMetadata = rendering as Record<string, unknown>;
  const assignment = String(renderingMetadata.patient_assignment ?? "");
  if (assignment === "assigned") {
    return "Assigned PHI";
  }
  if (assignment === "unassigned_generalized") {
    return "Generalized";
  }
  const renderMode = String(renderingMetadata.render_mode ?? "");
  return renderMode ? renderMode.replace(/_/g, " ") : null;
}

function searchAccessLabel(
  response: SearchResponse | null,
  hits: SearchHit[],
  activeRole: RoleProfile,
) {
  if (!response) {
    return activeRole.displayMode;
  }

  const assignmentModes = new Set(
    hits
      .map((hit) => hit.retrieval.rendering)
      .filter((rendering): rendering is Record<string, unknown> => Boolean(rendering))
      .map((rendering) => String(rendering.patient_assignment ?? "")),
  );

  if (assignmentModes.has("unassigned_generalized")) {
    return assignmentModes.has("assigned") ? "Mixed: assigned + generalized" : "Generalized";
  }
  if (assignmentModes.has("assigned")) {
    return "Full PHI: assigned";
  }

  const renderMode = response.pipeline.rendering?.render_mode;
  if (renderMode === "de_identified") {
    return "De-identified";
  }
  if (renderMode === "metadata_only") {
    return "Metadata only";
  }
  if (renderMode === "limited") {
    return "Limited clinical";
  }
  if (renderMode === "full_phi") {
    return activeRole.displayMode;
  }
  return renderMode?.replace(/_/g, " ") ?? activeRole.displayMode;
}

function matchesAuditSearch(event: AuditEvent, searchTerm: string) {
  const needle = searchTerm.trim().toLowerCase();
  if (!needle) {
    return true;
  }

  const haystack = [
    event.id,
    event.actor_display_name,
    event.actor_email,
    event.action,
    event.query_text,
    event.resource_type,
    event.resource_id,
    event.decision,
    event.ip_address,
    event.user_agent,
    ...event.role_snapshot,
    ...event.result_document_ids,
    JSON.stringify(event.metadata),
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  return haystack.includes(needle);
}

function buildAuditSummary(events: AuditEvent[]) {
  return {
    events: events.length,
    documents: new Set(events.flatMap((event) => event.result_document_ids)).size,
    denied: events.filter((event) => event.decision === "deny").length,
    phiDecrypts: events.filter((event) => event.action === "phi_decrypt").length,
  };
}

function buildIngestionSummary(documents: IngestionDocument[]) {
  return {
    documents: documents.length,
    extracted: documents.filter((document) => document.extraction_status === "extracted").length,
    reviewPending: documents.filter((document) => document.review_status === "pending").length,
    indexed: documents.filter((document) => document.status === "indexed").length,
  };
}

function buildUploadCandidate(file: File, index: number): UploadCandidate {
  const relativePath = browserRelativePath(file);
  const parts = relativePath.split(/[\\/]/).filter(Boolean);
  const fileStem = file.name.replace(/\.[^.]+$/, "");
  const nameParts = fileStem.split("-").map((part) => normalizeDatasetName(part));
  const patientName = nameParts[0] || `Dataset Patient ${index + 1}`;
  const physician = nameParts[1] || "Records Intake";
  const typedIndex = parts.findIndex((part) => part.toLowerCase() === "typed");
  const isHandwritten = parts.some((part) => part.toLowerCase().includes("handwritten"));
  const hospitalFolder = typedIndex >= 0 ? parts[typedIndex + 1] : nameParts[2];
  const hospital = isHandwritten
    ? "Handwritten Review Set"
    : normalizeDatasetName(hospitalFolder || "Dataset Hospital");
  const datasetKind = isHandwritten ? "handwritten" : "typed";

  return {
    file,
    relativePath,
    patientRef: `PATIENT_REF_${slugForIdentifier(patientName).slice(0, 42)}`,
    visitId: `VISIT_${stableShortHash(relativePath)}`,
    documentType: isHandwritten ? "handwritten_note" : "typed_pdf",
    hospital,
    physician,
    department: isHandwritten ? "Records Review" : "Clinical Records",
    metadata: {
      dataset_path: relativePath,
      dataset_kind: datasetKind,
      patient_name_hint: patientName,
      physician_name_hint: physician,
      upload_source: "ingestion_dashboard",
    },
  };
}

function browserRelativePath(file: File) {
  const browserFile = file as File & { webkitRelativePath?: string };
  return browserFile.webkitRelativePath || file.name;
}

function uniqueUploadCandidates(files: File[]) {
  const seen = new Set<string>();
  return files.filter((file) => {
    const key = `${browserRelativePath(file)}:${file.size}:${file.lastModified}`;
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

function normalizeDatasetName(value: string | undefined) {
  return (value ?? "")
    .replace(/\.[^.]+$/, "")
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function slugForIdentifier(value: string) {
  const slug = value
    .toUpperCase()
    .replace(/[^A-Z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return slug || "UNKNOWN";
}

function stableShortHash(value: string) {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) >>> 0;
  }
  return hash.toString(16).toUpperCase().padStart(8, "0");
}

function formatPercent(value: number | null) {
  if (value === null || Number.isNaN(value)) {
    return "confidence pending";
  }
  return `${Math.round(value * 100)}% confidence`;
}

function metadataString(event: AuditEvent, key: string) {
  const value = event.metadata[key];
  return typeof value === "string" && value ? value : "not recorded";
}

function metadataStringList(event: AuditEvent, key: string) {
  const value = event.metadata[key];
  if (Array.isArray(value)) {
    return value.map((item) => String(item)).filter(Boolean);
  }
  if (typeof value === "string" && value) {
    return [value];
  }
  return [];
}

function metadataMaskingSummary(event: AuditEvent) {
  const maskingModes = metadataStringList(event, "masking_modes");
  if (maskingModes.length) {
    return maskingModes.map((value) => value.replace(/_/g, " ")).join(" + ");
  }
  return metadataString(event, "masking_mode").replace(/_/g, " ");
}

function metadataAssignmentSummary(event: AuditEvent) {
  const assignmentModes = metadataStringList(event, "patient_assignment_modes");
  if (!assignmentModes.length) {
    return "assignment not recorded";
  }
  return assignmentModes.map((value) => value.replace(/_/g, " ")).join(" + ");
}

function metadataFilterSummary(event: AuditEvent) {
  const filters = event.metadata.filters;
  if (!filters || typeof filters !== "object") {
    return "filters not recorded";
  }
  const filterObject = filters as Record<string, unknown>;
  const sensitivity = filterObject.allowed_sensitivity_levels;
  if (Array.isArray(sensitivity)) {
    return `sensitivity=${sensitivity.join(",")}`;
  }
  return "filters recorded";
}

export default App;
