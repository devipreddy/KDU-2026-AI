from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.services.chunking import TokenChunker
from app.services.embedding_service import EmbeddingService
from app.services.enrichment_service import EnrichmentService
from app.services.job_queue_service import JobQueueService
from app.services.processing_service import ProcessingService
from app.services.processors.audio_processor import AudioProcessor
from app.services.processors.image_processor import ImageProcessor
from app.services.processors.pdf_processor import PDFProcessor
from app.services.providers.openai_provider import OpenAIProvider
from app.services.providers.speaker_diarization_provider import SpeakerDiarizer
from app.services.providers.whisper_provider import WhisperTranscriber
from app.services.reporting_service import ReportingService
from app.services.search_service import SearchService
from app.services.storage import StorageService
from app.services.text_normalizer import TextNormalizer
from app.services.vector_store import VectorStore


@lru_cache(maxsize=1)
def get_provider() -> OpenAIProvider:
    return OpenAIProvider(get_settings())


@lru_cache(maxsize=1)
def get_whisper_transcriber() -> WhisperTranscriber:
    return WhisperTranscriber(get_settings())


@lru_cache(maxsize=1)
def get_speaker_diarizer() -> SpeakerDiarizer:
    return SpeakerDiarizer(get_settings())


@lru_cache(maxsize=1)
def get_chunker() -> TokenChunker:
    return TokenChunker(get_settings().openai_generation_model)


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore(get_settings())


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(provider=get_provider(), vector_store=get_vector_store())


@lru_cache(maxsize=1)
def get_enrichment_service() -> EnrichmentService:
    return EnrichmentService(
        settings=get_settings(),
        provider=get_provider(),
        chunker=get_chunker(),
    )


@lru_cache(maxsize=1)
def get_processing_service() -> ProcessingService:
    settings = get_settings()
    return ProcessingService(
        settings=settings,
        pdf_processor=PDFProcessor(settings=settings, provider=get_provider()),
        image_processor=ImageProcessor(settings=settings, provider=get_provider()),
        audio_processor=AudioProcessor(
            transcriber=get_whisper_transcriber(),
            diarizer=get_speaker_diarizer(),
        ),
        normalizer=TextNormalizer(),
        chunker=get_chunker(),
        enrichment_service=get_enrichment_service(),
        embedding_service=get_embedding_service(),
        storage_service=StorageService(settings),
    )


@lru_cache(maxsize=1)
def get_search_service() -> SearchService:
    return SearchService(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )


@lru_cache(maxsize=1)
def get_reporting_service() -> ReportingService:
    return ReportingService()


@lru_cache(maxsize=1)
def get_job_queue_service() -> JobQueueService:
    return JobQueueService(settings=get_settings(), processing_service=get_processing_service())
