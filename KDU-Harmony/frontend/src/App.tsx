import { FormEvent, useMemo, useState } from "react";

import "./styles.css";

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

type ViewId = "search" | "audit";
type RoleId = "doctor" | "researcher" | "admin";
type AuditRoleId = RoleId | "nurse" | "records_staff";
type DocumentKind = "clinical_note" | "discharge_summary" | "lab_report" | "prescription";
type Sensitivity = "low" | "medium" | "high" | "restricted";
type ConfidenceLevel = "high" | "medium" | "low";
type AuditAction = "query_run" | "document_read" | "phi_decrypt" | "access_denied";
type AccessDecision = "allow" | "deny" | "review";

type RoleProfile = {
  label: string;
  email: string;
  displayMode: "Full PHI" | "De-identified" | "Metadata only";
  visibility: "full" | "de_identified" | "metadata_only";
};

type SearchResult = {
  id: string;
  patientFull: string;
  patientDeidentified: string;
  documentId: string;
  sourceDocument: string;
  documentKind: DocumentKind;
  hospital: string;
  physician: string;
  visitDate: string;
  section: string;
  page: number;
  sensitivity: Sensitivity;
  diagnosis: string;
  icdCodes: string[];
  confidence: number;
  confidenceLevel: ConfidenceLevel;
  snippetFull: string;
  snippetDeidentified: string;
  parentFull: string;
  parentDeidentified: string;
  matchedTerms: string[];
  retrievalSources: string[];
  searchCorpus: string;
};

type AuditEvent = {
  id: string;
  occurredAt: string;
  actorName: string;
  actorEmail: string;
  role: AuditRoleId;
  action: AuditAction;
  query: string;
  resourceType: "retrieval_query" | "document" | "phi_mapping";
  resourceId: string;
  documentIds: string[];
  chunkIds: string[];
  patientRef: string;
  maskingMode: "full_phi" | "de_identified" | "metadata_only" | "blocked";
  decision: AccessDecision;
  ipAddress: string;
  userAgent: string;
  filters: string[];
};

const roleProfiles: Record<RoleId, RoleProfile> = {
  doctor: {
    label: "Doctor",
    email: "doctor@example.com",
    displayMode: "Full PHI",
    visibility: "full",
  },
  researcher: {
    label: "Researcher",
    email: "researcher@example.com",
    displayMode: "De-identified",
    visibility: "de_identified",
  },
  admin: {
    label: "Admin",
    email: "admin@example.com",
    displayMode: "Metadata only",
    visibility: "metadata_only",
  },
};

const auditRoleLabels: Record<AuditRoleId, string> = {
  doctor: "Doctor",
  nurse: "Nurse",
  admin: "Admin",
  researcher: "Researcher",
  records_staff: "Records staff",
};

const auditActionLabels: Record<AuditAction, string> = {
  query_run: "Query run",
  document_read: "Document read",
  phi_decrypt: "PHI decrypt",
  access_denied: "Access denied",
};

const maskingModeLabels: Record<AuditEvent["maskingMode"], string> = {
  full_phi: "Full PHI",
  de_identified: "De-identified",
  metadata_only: "Metadata only",
  blocked: "Blocked",
};

const results: SearchResult[] = [
  {
    id: "chunk-5101",
    patientFull: "John Smith",
    patientDeidentified: "DEID-74A91C20",
    documentId: "DOC-CARD-0101",
    sourceDocument: "cardiology-follow-up.pdf",
    documentKind: "clinical_note",
    hospital: "Harmony General Hospital",
    physician: "Dr. Asha Raman",
    visitDate: "2025-02-14",
    section: "Assessment",
    page: 3,
    sensitivity: "high",
    diagnosis: "Atrial fibrillation",
    icdCodes: ["I48.91"],
    confidence: 0.94,
    confidenceLevel: "high",
    snippetFull:
      "John Smith presented with palpitations and intermittent chest pressure. Assessment notes atrial fibrillation with rate control improved on metoprolol.",
    snippetDeidentified:
      "DEID-74A91C20 presented with palpitations and intermittent chest pressure. Assessment notes atrial fibrillation with rate control improved on metoprolol.",
    parentFull:
      "Assessment: John Smith has atrial fibrillation with improved rate control. Treatment plan continues metoprolol 25 mg BID and cardiology follow-up in four weeks.",
    parentDeidentified:
      "Assessment: DEID-74A91C20 has atrial fibrillation with improved rate control. Treatment plan continues metoprolol 25 mg BID and cardiology follow-up in four weeks.",
    matchedTerms: ["cardiac", "Q1 2025", "atrial fibrillation", "metoprolol"],
    retrievalSources: ["BM25", "Vector", "Reranker"],
    searchCorpus:
      "cardiac q1 2025 atrial fibrillation palpitations metoprolol harmony general hospital cardiology i48.91 assessment",
  },
  {
    id: "chunk-5102",
    patientFull: "Maria Chen",
    patientDeidentified: "DEID-05FC8A11",
    documentId: "DOC-CARD-0228",
    sourceDocument: "discharge-summary-0228.pdf",
    documentKind: "discharge_summary",
    hospital: "North Valley Medical Center",
    physician: "Dr. Theo Klein",
    visitDate: "2025-03-03",
    section: "Discharge Summary",
    page: 1,
    sensitivity: "medium",
    diagnosis: "Stable angina",
    icdCodes: ["I20.9"],
    confidence: 0.89,
    confidenceLevel: "high",
    snippetFull:
      "Maria Chen was treated for exertional chest pain consistent with stable angina. Discharge plan includes aspirin therapy and outpatient stress testing.",
    snippetDeidentified:
      "DEID-05FC8A11 was treated for exertional chest pain consistent with stable angina. Discharge plan includes aspirin therapy and outpatient stress testing.",
    parentFull:
      "Discharge Summary: Maria Chen completed cardiac observation after exertional chest pain. No acute infarction was found. Follow-up is scheduled with cardiology.",
    parentDeidentified:
      "Discharge Summary: DEID-05FC8A11 completed cardiac observation after exertional chest pain. No acute infarction was found. Follow-up is scheduled with cardiology.",
    matchedTerms: ["cardiac", "Q1 2025", "stable angina", "aspirin"],
    retrievalSources: ["Vector", "Reranker"],
    searchCorpus:
      "cardiac q1 2025 stable angina chest pain aspirin discharge summary north valley medical center i20.9",
  },
  {
    id: "chunk-5103",
    patientFull: "Robert Diaz",
    patientDeidentified: "DEID-198C77E3",
    documentId: "DOC-ENDO-0312",
    sourceDocument: "lab-report-0312.pdf",
    documentKind: "lab_report",
    hospital: "Harmony General Hospital",
    physician: "Dr. Priya Nair",
    visitDate: "2025-04-12",
    section: "Lab Results",
    page: 2,
    sensitivity: "medium",
    diagnosis: "Type 2 diabetes mellitus",
    icdCodes: ["E11.9"],
    confidence: 0.76,
    confidenceLevel: "medium",
    snippetFull:
      "Robert Diaz has elevated A1C and fasting glucose. Lab interpretation supports ongoing type 2 diabetes management with metformin.",
    snippetDeidentified:
      "DEID-198C77E3 has elevated A1C and fasting glucose. Lab interpretation supports ongoing type 2 diabetes management with metformin.",
    parentFull:
      "Lab Results: Robert Diaz showed A1C 8.1%, fasting glucose 166 mg/dL, and normal creatinine. The note recommends medication adherence review.",
    parentDeidentified:
      "Lab Results: DEID-198C77E3 showed A1C 8.1%, fasting glucose 166 mg/dL, and normal creatinine. The note recommends medication adherence review.",
    matchedTerms: ["diabetes", "A1C", "metformin"],
    retrievalSources: ["BM25"],
    searchCorpus:
      "endocrine diabetes a1c metformin lab report harmony general hospital e11.9 glucose",
  },
  {
    id: "chunk-5104",
    patientFull: "Evelyn Brooks",
    patientDeidentified: "DEID-AB31E042",
    documentId: "DOC-RENAL-0117",
    sourceDocument: "nephrology-note-scan.pdf",
    documentKind: "clinical_note",
    hospital: "Lakeview Regional Hospital",
    physician: "Dr. Leena Kapoor",
    visitDate: "2025-01-17",
    section: "Treatment Plan",
    page: 5,
    sensitivity: "restricted",
    diagnosis: "Chronic kidney disease stage 3",
    icdCodes: ["N18.30"],
    confidence: 0.58,
    confidenceLevel: "low",
    snippetFull:
      "Evelyn Brooks has CKD stage 3 with medication reconciliation pending. OCR confidence is reduced because the source page is a scanned note.",
    snippetDeidentified:
      "DEID-AB31E042 has CKD stage 3 with medication reconciliation pending. OCR confidence is reduced because the source page is a scanned note.",
    parentFull:
      "Treatment Plan: Evelyn Brooks should repeat renal panel in six weeks. Avoid nephrotoxic medications and review blood pressure log at follow-up.",
    parentDeidentified:
      "Treatment Plan: DEID-AB31E042 should repeat renal panel in six weeks. Avoid nephrotoxic medications and review blood pressure log at follow-up.",
    matchedTerms: ["renal", "CKD", "OCR review"],
    retrievalSources: ["Vector"],
    searchCorpus:
      "renal kidney ckd chronic kidney disease scanned note ocr lakeview regional hospital n18.30 treatment plan",
  },
];

const auditEvents: AuditEvent[] = [
  {
    id: "AUD-9008",
    occurredAt: "2025-04-12T15:42:10Z",
    actorName: "Asha Raman",
    actorEmail: "doctor@example.com",
    role: "doctor",
    action: "phi_decrypt",
    query: "resolve patient token for active cardiology follow-up",
    resourceType: "phi_mapping",
    resourceId: "PHI-MAP-74A91C20",
    documentIds: ["DOC-CARD-0101"],
    chunkIds: ["chunk-5101"],
    patientRef: "PATIENT_REF_42",
    maskingMode: "full_phi",
    decision: "allow",
    ipAddress: "10.0.8.14",
    userAgent: "HarmonyWeb/0.1",
    filters: ["role=doctor", "active_treatment=true", "sensitivity<=high"],
  },
  {
    id: "AUD-9007",
    occurredAt: "2025-04-12T15:41:55Z",
    actorName: "Asha Raman",
    actorEmail: "doctor@example.com",
    role: "doctor",
    action: "document_read",
    query: "patients with cardiac issues treated in Q1 2025",
    resourceType: "document",
    resourceId: "DOC-CARD-0101",
    documentIds: ["DOC-CARD-0101"],
    chunkIds: ["chunk-5101"],
    patientRef: "PATIENT_REF_42",
    maskingMode: "full_phi",
    decision: "allow",
    ipAddress: "10.0.8.14",
    userAgent: "HarmonyWeb/0.1",
    filters: ["diagnosis_category=cardiac", "date=2025-Q1", "role=doctor"],
  },
  {
    id: "AUD-9006",
    occurredAt: "2025-04-12T15:41:48Z",
    actorName: "Asha Raman",
    actorEmail: "doctor@example.com",
    role: "doctor",
    action: "query_run",
    query: "patients with cardiac issues treated in Q1 2025",
    resourceType: "retrieval_query",
    resourceId: "RET-20250412-154148",
    documentIds: ["DOC-CARD-0101", "DOC-CARD-0228"],
    chunkIds: ["chunk-5101", "chunk-5102"],
    patientRef: "multiple",
    maskingMode: "full_phi",
    decision: "allow",
    ipAddress: "10.0.8.14",
    userAgent: "HarmonyWeb/0.1",
    filters: ["diagnosis_category=cardiac", "date=2025-Q1", "role=doctor"],
  },
  {
    id: "AUD-9005",
    occurredAt: "2025-04-12T14:28:36Z",
    actorName: "Noor Patel",
    actorEmail: "researcher@example.com",
    role: "researcher",
    action: "query_run",
    query: "cardiac outcomes after beta blocker therapy",
    resourceType: "retrieval_query",
    resourceId: "RET-20250412-142836",
    documentIds: ["DOC-CARD-0101", "DOC-CARD-0228"],
    chunkIds: ["chunk-5101", "chunk-5102"],
    patientRef: "de-identified cohort",
    maskingMode: "de_identified",
    decision: "allow",
    ipAddress: "10.0.12.22",
    userAgent: "HarmonyWeb/0.1",
    filters: ["role=researcher", "phi_visibility=de_identified", "cohort=cardiac"],
  },
  {
    id: "AUD-9004",
    occurredAt: "2025-04-12T13:05:09Z",
    actorName: "Mina Ortiz",
    actorEmail: "records@example.com",
    role: "records_staff",
    action: "document_read",
    query: "retrieve discharge summary for records reconciliation",
    resourceType: "document",
    resourceId: "DOC-CARD-0228",
    documentIds: ["DOC-CARD-0228"],
    chunkIds: ["chunk-5102"],
    patientRef: "PATIENT_REF_51",
    maskingMode: "metadata_only",
    decision: "allow",
    ipAddress: "10.0.6.33",
    userAgent: "HarmonyWeb/0.1",
    filters: ["role=records_staff", "document_type=discharge_summary"],
  },
  {
    id: "AUD-9003",
    occurredAt: "2025-04-12T11:19:42Z",
    actorName: "Theo Klein",
    actorEmail: "admin@example.com",
    role: "admin",
    action: "query_run",
    query: "restricted nephrology scanned notes with low OCR confidence",
    resourceType: "retrieval_query",
    resourceId: "RET-20250412-111942",
    documentIds: ["DOC-RENAL-0117"],
    chunkIds: ["chunk-5104"],
    patientRef: "hidden",
    maskingMode: "metadata_only",
    decision: "review",
    ipAddress: "10.0.5.18",
    userAgent: "HarmonyWeb/0.1",
    filters: ["role=admin", "sensitivity=restricted", "ocr_confidence<0.65"],
  },
  {
    id: "AUD-9002",
    occurredAt: "2025-04-12T10:33:20Z",
    actorName: "Elena Moore",
    actorEmail: "nurse@example.com",
    role: "nurse",
    action: "access_denied",
    query: "Evelyn Brooks nephrology restricted scan",
    resourceType: "document",
    resourceId: "DOC-RENAL-0117",
    documentIds: ["DOC-RENAL-0117"],
    chunkIds: [],
    patientRef: "PATIENT_REF_84",
    maskingMode: "blocked",
    decision: "deny",
    ipAddress: "10.0.9.44",
    userAgent: "HarmonyWeb/0.1",
    filters: ["role=nurse", "sensitivity=restricted", "deny_reason=no_treatment_relationship"],
  },
  {
    id: "AUD-9001",
    occurredAt: "2025-04-12T09:12:04Z",
    actorName: "Priya Nair",
    actorEmail: "doctor@example.com",
    role: "doctor",
    action: "document_read",
    query: "diabetes A1C metformin lab reports",
    resourceType: "document",
    resourceId: "DOC-ENDO-0312",
    documentIds: ["DOC-ENDO-0312"],
    chunkIds: ["chunk-5103"],
    patientRef: "PATIENT_REF_63",
    maskingMode: "full_phi",
    decision: "allow",
    ipAddress: "10.0.8.19",
    userAgent: "HarmonyWeb/0.1",
    filters: ["role=doctor", "diagnosis=diabetes", "document_type=lab_report"],
  },
];

const documentKindLabels: Record<DocumentKind, string> = {
  clinical_note: "Clinical note",
  discharge_summary: "Discharge summary",
  lab_report: "Lab report",
  prescription: "Prescription",
};

const sensitivityLabels: Record<Sensitivity, string> = {
  low: "Low",
  medium: "Medium",
  high: "High",
  restricted: "Restricted",
};

const stopwords = new Set([
  "and",
  "for",
  "from",
  "in",
  "of",
  "patients",
  "records",
  "show",
  "the",
  "treated",
  "with",
]);

const auditDateFormatter = new Intl.DateTimeFormat("en-US", {
  dateStyle: "medium",
  timeStyle: "short",
});

function App() {
  const [activeView, setActiveView] = useState<ViewId>("search");
  const [query, setQuery] = useState("patients with cardiac issues treated in Q1 2025");
  const [submittedQuery, setSubmittedQuery] = useState(query);
  const [role, setRole] = useState<RoleId>("doctor");
  const [documentKind, setDocumentKind] = useState<"all" | DocumentKind>("all");
  const [hospital, setHospital] = useState("all");
  const [sensitivity, setSensitivity] = useState<"all" | Sensitivity>("all");
  const [minimumConfidence, setMinimumConfidence] = useState(55);
  const [auditSearch, setAuditSearch] = useState("");
  const [auditRole, setAuditRole] = useState<"all" | AuditRoleId>("all");
  const [auditAction, setAuditAction] = useState<"all" | AuditAction>("all");
  const [auditDecision, setAuditDecision] = useState<"all" | AccessDecision>("all");

  const activeRole = roleProfiles[role];
  const hospitals = useMemo(
    () => Array.from(new Set(results.map((result) => result.hospital))).sort(),
    [],
  );

  const filteredResults = useMemo(
    () =>
      results.filter((result) => {
        if (!matchesQuery(result, submittedQuery)) {
          return false;
        }
        if (documentKind !== "all" && result.documentKind !== documentKind) {
          return false;
        }
        if (hospital !== "all" && result.hospital !== hospital) {
          return false;
        }
        if (sensitivity !== "all" && result.sensitivity !== sensitivity) {
          return false;
        }
        return result.confidence * 100 >= minimumConfidence;
      }),
    [documentKind, hospital, minimumConfidence, sensitivity, submittedQuery],
  );

  const filteredAuditEvents = useMemo(
    () =>
      auditEvents
        .filter((event) => {
          if (auditRole !== "all" && event.role !== auditRole) {
            return false;
          }
          if (auditAction !== "all" && event.action !== auditAction) {
            return false;
          }
          if (auditDecision !== "all" && event.decision !== auditDecision) {
            return false;
          }
          return matchesAuditSearch(event, auditSearch);
        })
        .sort(
          (left, right) =>
            new Date(right.occurredAt).getTime() - new Date(left.occurredAt).getTime(),
        ),
    [auditAction, auditDecision, auditRole, auditSearch],
  );

  const auditSummary = useMemo(() => buildAuditSummary(filteredAuditEvents), [filteredAuditEvents]);
  const topResult = filteredResults[0];

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmittedQuery(query);
  }

  function resetFilters() {
    setDocumentKind("all");
    setHospital("all");
    setSensitivity("all");
    setMinimumConfidence(55);
  }

  function resetAuditFilters() {
    setAuditSearch("");
    setAuditRole("all");
    setAuditAction("all");
    setAuditDecision("all");
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

      {activeView === "search" ? (
        <section className="search-workspace" aria-label="Search workspace">
          <aside className="filter-panel" aria-label="Search filters">
            <div className="panel-heading">
              <span>Role</span>
              <strong>{activeRole.displayMode}</strong>
            </div>

            <div className="role-segment" aria-label="Role selector">
              {(Object.keys(roleProfiles) as RoleId[]).map((roleId) => (
                <button
                  className={role === roleId ? "selected" : ""}
                  key={roleId}
                  onClick={() => setRole(roleId)}
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
                onChange={(event) => setDocumentKind(event.target.value as "all" | DocumentKind)}
              >
                <option value="all">All types</option>
                {Object.entries(documentKindLabels).map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
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
              <span>Sensitivity</span>
              <select
                value={sensitivity}
                onChange={(event) => setSensitivity(event.target.value as "all" | Sensitivity)}
              >
                <option value="all">All levels</option>
                {Object.entries(sensitivityLabels).map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </label>

            <label className="field">
              <span>Minimum confidence</span>
              <input
                max="95"
                min="40"
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

          <section className="results-area" aria-label="Search results">
            <form className="query-panel" onSubmit={handleSubmit}>
              <label htmlFor="query">Natural language query</label>
              <div className="query-row">
                <textarea
                  id="query"
                  onChange={(event) => setQuery(event.target.value)}
                  rows={3}
                  value={query}
                />
                <button type="submit">Search</button>
              </div>
            </form>

            <div className="result-summary" aria-live="polite">
              <div>
                <span>{filteredResults.length} results</span>
                <strong>{submittedQuery}</strong>
              </div>
              <div className={`masking-mode ${activeRole.visibility}`}>
                {activeRole.displayMode}
              </div>
            </div>

            <div className="content-grid">
              <section className="result-list" aria-label="Result list">
                {filteredResults.map((result) => (
                  <ResultCard key={result.id} result={result} role={role} />
                ))}

                {filteredResults.length === 0 ? (
                  <div className="empty-state">
                    <strong>No matching records</strong>
                    <span>Adjust the query or filters.</span>
                  </div>
                ) : null}
              </section>

              <aside className="detail-panel" aria-label="Selected result context">
                <div className="panel-heading">
                  <span>Top context</span>
                  <strong>{topResult ? topResult.documentId : "No result"}</strong>
                </div>
                {topResult ? <ContextPreview result={topResult} role={role} /> : null}
              </aside>
            </div>
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
              <select
                value={auditRole}
                onChange={(event) => setAuditRole(event.target.value as "all" | AuditRoleId)}
              >
                <option value="all">All roles</option>
                {Object.entries(auditRoleLabels).map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </label>

            <label className="field compact">
              <span>Action</span>
              <select
                value={auditAction}
                onChange={(event) => setAuditAction(event.target.value as "all" | AuditAction)}
              >
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
                onChange={(event) => setAuditDecision(event.target.value as "all" | AccessDecision)}
              >
                <option value="all">All decisions</option>
                <option value="allow">Allow</option>
                <option value="deny">Deny</option>
                <option value="review">Review</option>
              </select>
            </label>

            <button
              className="secondary-action audit-reset"
              onClick={resetAuditFilters}
              type="button"
            >
              Reset
            </button>
          </section>

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
                <strong>{filteredAuditEvents.length} matching events</strong>
              </div>
              <div className="audit-freshness">Immutable append-only stream</div>
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
                  {filteredAuditEvents.map((event) => (
                    <AuditRow event={event} key={event.id} />
                  ))}

                  {filteredAuditEvents.length === 0 ? (
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

function ResultCard({ result, role }: { result: SearchResult; role: RoleId }) {
  const isMetadataOnly = roleProfiles[role].visibility === "metadata_only";
  const patientLabel = patientDisplay(result, role);

  return (
    <article className="result-card">
      <div className="result-card-header">
        <div>
          <span className="result-type">{documentKindLabels[result.documentKind]}</span>
          <h2>{isMetadataOnly ? result.documentId : result.diagnosis}</h2>
        </div>
        <ConfidenceIndicator level={result.confidenceLevel} score={result.confidence} />
      </div>

      <dl className="metadata-strip">
        <div>
          <dt>Patient</dt>
          <dd>{patientLabel}</dd>
        </div>
        <div>
          <dt>Visit</dt>
          <dd>{result.visitDate}</dd>
        </div>
        <div>
          <dt>Section</dt>
          <dd>{result.section}</dd>
        </div>
        <div>
          <dt>Sensitivity</dt>
          <dd>{sensitivityLabels[result.sensitivity]}</dd>
        </div>
      </dl>

      {isMetadataOnly ? (
        <div className="metadata-only-view">
          <span>{result.hospital}</span>
          <span>{result.physician}</span>
          <span>{result.icdCodes.join(", ")}</span>
        </div>
      ) : (
        <p className="snippet">
          {role === "doctor" ? result.snippetFull : result.snippetDeidentified}
        </p>
      )}

      <div className="terms-row">
        {result.matchedTerms.map((term) => (
          <span key={term}>{term}</span>
        ))}
      </div>

      <footer className="citation-row">
        <span>
          {result.documentId} | p. {result.page} | {result.section}
        </span>
        <span>{result.retrievalSources.join(" + ")}</span>
      </footer>
    </article>
  );
}

function ContextPreview({ result, role }: { result: SearchResult; role: RoleId }) {
  const isMetadataOnly = roleProfiles[role].visibility === "metadata_only";

  return (
    <div className="context-preview">
      <dl>
        <div>
          <dt>Source</dt>
          <dd>{result.sourceDocument}</dd>
        </div>
        <div>
          <dt>Hospital</dt>
          <dd>{result.hospital}</dd>
        </div>
        <div>
          <dt>Physician</dt>
          <dd>{result.physician}</dd>
        </div>
        <div>
          <dt>ICD</dt>
          <dd>{result.icdCodes.join(", ")}</dd>
        </div>
      </dl>

      {isMetadataOnly ? (
        <p className="metadata-note">Clinical text hidden for this role.</p>
      ) : (
        <p>{role === "doctor" ? result.parentFull : result.parentDeidentified}</p>
      )}
    </div>
  );
}

function AuditRow({ event }: { event: AuditEvent }) {
  return (
    <tr>
      <td>
        <strong>{auditDateFormatter.format(new Date(event.occurredAt))}</strong>
        <span>{event.id}</span>
      </td>
      <td>
        <strong>{event.actorName}</strong>
        <span>{event.actorEmail}</span>
        <span>{auditRoleLabels[event.role]}</span>
      </td>
      <td>
        <span className="action-chip">{auditActionLabels[event.action]}</span>
      </td>
      <td>
        <strong>{event.resourceId}</strong>
        <span>{event.resourceType}</span>
        <span>{event.documentIds.join(", ")}</span>
        {event.chunkIds.length > 0 ? <span>{event.chunkIds.join(", ")}</span> : null}
      </td>
      <td className="query-cell">
        <strong>{event.query}</strong>
        <span>{event.patientRef}</span>
        <span>{event.filters.join(" | ")}</span>
      </td>
      <td>
        <span className={`decision-pill ${event.decision}`}>{event.decision}</span>
        <span>{event.ipAddress}</span>
      </td>
      <td>
        <strong>{maskingModeLabels[event.maskingMode]}</strong>
        <span>{event.userAgent}</span>
      </td>
    </tr>
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

function patientDisplay(result: SearchResult, role: RoleId) {
  const visibility = roleProfiles[role].visibility;
  if (visibility === "full") {
    return result.patientFull;
  }
  if (visibility === "metadata_only") {
    return "Hidden";
  }
  return result.patientDeidentified;
}

function matchesQuery(result: SearchResult, query: string) {
  const terms = query
    .toLowerCase()
    .replace(/[^a-z0-9.\s-]/g, " ")
    .split(/\s+/)
    .filter((term) => term.length > 1 && !stopwords.has(term));

  if (terms.length === 0) {
    return true;
  }

  const searchable =
    `${result.searchCorpus} ${result.diagnosis} ${result.sourceDocument}`.toLowerCase();
  const matches = terms.filter((term) => searchable.includes(term));
  return matches.length >= Math.min(2, terms.length);
}

function matchesAuditSearch(event: AuditEvent, searchTerm: string) {
  const needle = searchTerm.trim().toLowerCase();
  if (!needle) {
    return true;
  }

  const haystack = [
    event.id,
    event.actorName,
    event.actorEmail,
    auditRoleLabels[event.role],
    auditActionLabels[event.action],
    event.query,
    event.resourceType,
    event.resourceId,
    event.patientRef,
    event.maskingMode,
    event.decision,
    event.ipAddress,
    event.userAgent,
    ...event.documentIds,
    ...event.chunkIds,
    ...event.filters,
  ]
    .join(" ")
    .toLowerCase();

  return haystack.includes(needle);
}

function buildAuditSummary(events: AuditEvent[]) {
  return {
    events: events.length,
    documents: new Set(events.flatMap((event) => event.documentIds)).size,
    denied: events.filter((event) => event.decision === "deny").length,
    phiDecrypts: events.filter((event) => event.action === "phi_decrypt").length,
  };
}

export default App;
