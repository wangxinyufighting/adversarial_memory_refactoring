import enum


if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        """Small Python 3.10 compatibility shim for libraries expecting enum.StrEnum."""

        def __str__(self) -> str:
            return self.value

    enum.StrEnum = StrEnum
