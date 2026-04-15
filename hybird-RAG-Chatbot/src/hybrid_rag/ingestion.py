from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - compatibility fallback
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .config import Settings
from .utils import content_hash, normalize_whitespace, read_json_file, write_json_file


class KnowledgeBase:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        Path(self.settings.chroma_dir).mkdir(parents=True, exist_ok=True)
        self.docstore_path = self.settings.docstore_path
        self.registry = read_json_file(self.docstore_path, default=[])
        self.embeddings = OpenAIEmbeddings(
            model=self.settings.openrouter_embedding_model,
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.openrouter_base_url,
            default_headers={
                "HTTP-Referer": self.settings.openrouter_site_url,
                "X-Title": self.settings.openrouter_app_name,
            },
        )
        self.vectorstore = Chroma(
            collection_name=self.settings.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.settings.chroma_dir,
        )

    def _load_pdf_documents(self, uploaded_files: list[Any]) -> list[Document]:
        documents: list[Document] = []
        for uploaded_file in uploaded_files:
            suffix = Path(uploaded_file.name).suffix or ".pdf"
            with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name
            loader = PyPDFLoader(temp_path)
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = uploaded_file.name
                page.metadata["source_type"] = "pdf"
            documents.extend(pages)
        return documents

    def _load_url_documents(self, raw_urls: str) -> list[Document]:
        documents: list[Document] = []
        urls = [line.strip() for line in raw_urls.splitlines() if line.strip()]
        for url in urls:
            loader = WebBaseLoader(web_paths=[url])
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source"] = url
                doc.metadata["source_type"] = "url"
            documents.extend(loaded)
        return documents

    def ingest_api_source(self, source_type: str, source_path: str) -> dict[str, int]:
        if source_type == "pdf":
            local_path = Path(source_path)
            if not local_path.exists():
                raise FileNotFoundError(f"PDF not found: {source_path}")
            loader = PyPDFLoader(str(local_path))
            documents = loader.load()
            for page in documents:
                page.metadata["source"] = local_path.name
                page.metadata["source_type"] = "pdf"
        elif source_type == "url":
            documents = self._load_url_documents(source_path)
        else:
            raise ValueError("source_type must be either 'pdf' or 'url'")

        chunks = self._contextual_chunk(documents)
        existing_ids = {entry["chunk_id"] for entry in self.registry}
        new_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_id"] not in existing_ids]
        if new_chunks:
            self.vectorstore.add_documents(new_chunks, ids=[doc.metadata["chunk_id"] for doc in new_chunks])
            for chunk in new_chunks:
                self.registry.append(
                    {
                        "chunk_id": chunk.metadata["chunk_id"],
                        "source": chunk.metadata.get("source", "unknown"),
                        "source_type": chunk.metadata.get("source_type", "unknown"),
                        "chunk_index": chunk.metadata.get("chunk_index", 0),
                        "text": chunk.page_content,
                        "metadata": chunk.metadata,
                    }
                )
            write_json_file(self.docstore_path, self.registry)
        unique_sources = {doc.metadata.get("source", "unknown") for doc in documents}
        return {
            "status": "success",
            "documents_processed": len(unique_sources),
            "chunks_created": len(new_chunks),
        }

    def _split_into_sections(self, text: str) -> list[dict[str, str]]:
        sections: list[dict[str, str]] = []
        current_heading = "Document"
        buffer: list[str] = []
        for raw_part in text.split("\n\n"):
            part = normalize_whitespace(raw_part)
            if not part:
                continue
            is_heading = len(part) < 100 and part == part.title()
            if is_heading:
                if buffer:
                    sections.append({"heading": current_heading, "text": " ".join(buffer)})
                    buffer = []
                current_heading = part
                continue
            buffer.append(part)
            if len(" ".join(buffer)) >= self.settings.chunk_size * 2:
                sections.append({"heading": current_heading, "text": " ".join(buffer)})
                buffer = []
        if buffer:
            sections.append({"heading": current_heading, "text": " ".join(buffer)})
        if not sections and text.strip():
            sections.append({"heading": "Document", "text": normalize_whitespace(text)})
        return sections

    def _contextual_chunk(self, documents: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks: list[Document] = []
        for doc in documents:
            raw_text = doc.page_content.strip()
            if not raw_text:
                continue
            base_metadata = dict(doc.metadata)
            sections = self._split_into_sections(raw_text)
            chunk_index = 0
            for section in sections:
                splits = splitter.split_text(section["text"])
                for split in splits:
                    chunk_id = content_hash(f"{base_metadata.get('source','unknown')}::{chunk_index}::{split}")
                    metadata = {
                        **base_metadata,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                        "section": section["heading"],
                    }
                    chunks.append(Document(page_content=normalize_whitespace(split), metadata=metadata))
                    chunk_index += 1
        return chunks

    def ingest_sources(self, uploaded_files: list[Any], raw_urls: str) -> dict[str, int]:
        documents: list[Document] = []
        documents.extend(self._load_pdf_documents(uploaded_files))
        documents.extend(self._load_url_documents(raw_urls))

        if not documents:
            return {"status": "noop", "documents_processed": 0, "chunks_created": 0}

        chunks = self._contextual_chunk(documents)
        if not chunks:
            return {"status": "noop", "documents_processed": 0, "chunks_created": 0}

        existing_ids = {entry["chunk_id"] for entry in self.registry}
        new_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_id"] not in existing_ids]

        if new_chunks:
            self.vectorstore.add_documents(new_chunks, ids=[doc.metadata["chunk_id"] for doc in new_chunks])
            for chunk in new_chunks:
                self.registry.append(
                    {
                        "chunk_id": chunk.metadata["chunk_id"],
                        "source": chunk.metadata.get("source", "unknown"),
                        "source_type": chunk.metadata.get("source_type", "unknown"),
                        "chunk_index": chunk.metadata.get("chunk_index", 0),
                        "text": chunk.page_content,
                        "metadata": chunk.metadata,
                    }
                )
            write_json_file(self.docstore_path, self.registry)

        unique_sources = {doc.metadata.get("source", "unknown") for doc in documents}
        return {
            "status": "success",
            "documents_processed": len(unique_sources),
            "chunks_created": len(new_chunks),
        }

    def get_registry(self) -> list[dict[str, Any]]:
        self.registry = read_json_file(self.docstore_path, default=[])
        return self.registry

    def get_stats(self) -> dict[str, Any]:
        registry = self.get_registry()
        sources = sorted({item["source"] for item in registry})
        return {
            "documents": len(sources),
            "chunks": len(registry),
            "sources": sources,
        }

    def get_source_details(self) -> list[dict[str, Any]]:
        registry = self.get_registry()
        grouped: dict[str, dict[str, Any]] = {}
        for item in registry:
            source = item["source"]
            source_entry = grouped.setdefault(
                source,
                {
                    "source": source,
                    "source_type": item.get("source_type", "unknown"),
                    "chunks": 0,
                    "sections": set(),
                    "preview": item["text"][:220],
                },
            )
            source_entry["chunks"] += 1
            section = item.get("metadata", {}).get("section")
            if section:
                source_entry["sections"].add(section)
        details = []
        for value in grouped.values():
            details.append(
                {
                    "source": value["source"],
                    "source_type": value["source_type"],
                    "chunks": value["chunks"],
                    "sections": sorted(value["sections"]),
                    "preview": value["preview"],
                }
            )
        return sorted(details, key=lambda item: item["source"].lower())

    def remove_source(self, source_name: str) -> int:
        registry = self.get_registry()
        to_remove = [item for item in registry if item["source"] == source_name]
        if not to_remove:
            return 0
        ids = [item["chunk_id"] for item in to_remove]
        self.vectorstore.delete(ids=ids)
        self.registry = [item for item in registry if item["source"] != source_name]
        write_json_file(self.docstore_path, self.registry)
        return len(ids)

    def clear_all(self) -> int:
        registry = self.get_registry()
        ids = [item["chunk_id"] for item in registry]
        if ids:
            self.vectorstore.delete(ids=ids)
        self.registry = []
        write_json_file(self.docstore_path, self.registry)
        return len(ids)
