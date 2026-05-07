from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from sensenova_drone.observation import Observation


@dataclass
class MemoryEntry:
    observation: Observation
    latent: Any | None = None
    embedding: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RealObservationMemory:
    """
    Stores only real observations gathered from Gazebo/PX4 runtime.

    Generated futures are intentionally rejected to preserve the distinction
    between truth and imagined rollouts.
    """

    def __init__(self, entries: list[MemoryEntry] | None = None, max_size: int | None = None):
        self._entries: list[MemoryEntry] = []
        self.max_size = max_size

        for entry in entries or []:
            self.append(entry)

    def append(self, entry: MemoryEntry) -> None:
        from sensenova_drone.world_state import PredictedFuture

        if isinstance(entry, PredictedFuture):
            raise TypeError("PredictedFuture objects must never be stored in RealObservationMemory.")

        if not isinstance(entry, MemoryEntry):
            raise TypeError(f"RealObservationMemory accepts MemoryEntry objects, got {type(entry)!r}.")

        source = str(entry.metadata.get("source", "")).strip().lower()
        if source in {"generated_kairos_rollout", "predicted_future", "imagined"}:
            raise ValueError("Generated futures must never be appended as real observation memory.")

        if self.max_size is not None and self.max_size <= 0:
            return

        self._entries.append(entry)

        if self.max_size is not None and len(self._entries) > self.max_size:
            self._entries = self._entries[-self.max_size :]

    def extend(self, entries: list[MemoryEntry]) -> None:
        for entry in entries:
            self.append(entry)

    def recent(self, count: int) -> list[MemoryEntry]:
        if count <= 0:
            return []
        return self._entries[-count:]

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[MemoryEntry]:
        return iter(self._entries)

    def __getitem__(self, index: int) -> MemoryEntry:
        return self._entries[index]
