from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pydantic import BaseModel, Field


class TravelOffer(BaseModel):
    id: str
    title: str
    destination: str
    origin: str
    price_usd: int
    nights: int
    airline: str
    hotel: str
    highlights: list[str] = Field(default_factory=list)

    @property
    def price_display(self) -> str:
        return f"${self.price_usd:,}"


CATALOG: list[TravelOffer] = [
    TravelOffer(
        id="offer_paris_spring",
        title="Paris Spring Escape",
        destination="Paris",
        origin="New York",
        price_usd=1199,
        nights=5,
        airline="Air France",
        hotel="Le Marais Atelier",
        highlights=["Round-trip airfare", "Boutique hotel", "Seine dinner cruise"],
    ),
    TravelOffer(
        id="offer_tokyo_city",
        title="Tokyo City Lights",
        destination="Tokyo",
        origin="San Francisco",
        price_usd=1699,
        nights=6,
        airline="ANA",
        hotel="Shibuya Skyline Hotel",
        highlights=["Non-stop flight", "Breakfast included", "JR rail pass"],
    ),
    TravelOffer(
        id="offer_rome_weekend",
        title="Rome Weekend Classic",
        destination="Rome",
        origin="Boston",
        price_usd=1049,
        nights=4,
        airline="ITA Airways",
        hotel="Piazza Navona Suites",
        highlights=["Historic center stay", "Airport transfer", "Colosseum tour"],
    ),
    TravelOffer(
        id="offer_bali_retreat",
        title="Bali Ocean Retreat",
        destination="Bali",
        origin="Los Angeles",
        price_usd=1899,
        nights=7,
        airline="Singapore Airlines",
        hotel="Uluwatu Cove Resort",
        highlights=["Resort breakfast", "Spa credit", "Private airport pickup"],
    ),
    TravelOffer(
        id="offer_dubai_stopover",
        title="Dubai Skyline Stopover",
        destination="Dubai",
        origin="Chicago",
        price_usd=1399,
        nights=4,
        airline="Emirates",
        hotel="Downtown Pearl Dubai",
        highlights=["Burj view room", "Desert safari", "Late checkout"],
    ),
]


@dataclass(slots=True)
class TravelCatalog:
    offers: list[TravelOffer]

    def search(
        self,
        query: str,
        *,
        limit: int = 3,
    ) -> list[TravelOffer]:
        query_norm = query.lower().strip()
        if not query_norm:
            return self.offers[:limit]

        ranked = sorted(
            self.offers,
            key=lambda offer: self._score_offer(offer, query_norm),
            reverse=True,
        )
        return [offer for offer in ranked if self._score_offer(offer, query_norm) > 0][
            :limit
        ] or self.offers[:limit]

    def get(self, offer_id: str) -> TravelOffer | None:
        return next((offer for offer in self.offers if offer.id == offer_id), None)

    def _score_offer(self, offer: TravelOffer, query: str) -> int:
        haystack = " ".join(
            [
                offer.title,
                offer.destination,
                offer.origin,
                offer.airline,
                offer.hotel,
                " ".join(offer.highlights),
            ]
        ).lower()
        score = 0
        for token in query.split():
            if token in haystack:
                score += 1
        if offer.destination.lower() in query:
            score += 3
        return score


def build_catalog() -> TravelCatalog:
    return TravelCatalog(offers=CATALOG.copy())

