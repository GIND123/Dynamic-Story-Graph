"""Small shared typing helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | Mapping[str, "JSONValue"]
