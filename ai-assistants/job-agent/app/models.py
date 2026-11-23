from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Job:
    source: str            # djinni | dou | robota
    ext_id: str
    url: str
    title: str
    company: str = ""
    description: str = ""
    salary: str = ""
    location: str = ""
    posted_at: datetime | None = None
    company_slug: str = ""     # DOU company slug (for size lookup)
    company_size: int = 0      # max number of specialists, 0 = unknown
    extra: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.source}:{self.ext_id}"
