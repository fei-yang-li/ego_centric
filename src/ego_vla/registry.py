from __future__ import annotations

from typing import Callable, Generic, Iterator, TypeVar

T = TypeVar("T")

Factory = Callable[..., T]


class Registry(Generic[T]):
    """A name -> factory registry for pluggable pipeline backends.

    Each pipeline stage (calibration, pose, depth, ego-motion, objects,
    actions, ...) owns one registry. New backends register a factory under a
    short name and become selectable from config without editing the pipeline
    dispatch code.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Factory[T]] = {}

    def register(
        self,
        name: str,
        factory: Factory[T] | None = None,
        *,
        override: bool = False,
    ) -> Factory[T] | Callable[[Factory[T]], Factory[T]]:
        """Register a factory by name.

        Usable as a decorator (``@registry.register("foo")``) or directly
        (``registry.register("foo", factory)``).
        """

        key = name.lower()

        def _register(func: Factory[T]) -> Factory[T]:
            if key in self._factories and not override:
                raise ValueError(
                    f"{self.kind} backend already registered: {name!r}"
                )
            self._factories[key] = func
            return func

        if factory is not None:
            return _register(factory)
        return _register

    def create(self, name: str | None, *args: object, **kwargs: object) -> T:
        key = (name or "none").lower()
        if key not in self._factories:
            available = ", ".join(self.available()) or "(none registered)"
            raise ValueError(
                f"Unknown {self.kind} backend: {name!r}. Available: {available}"
            )
        return self._factories[key](*args, **kwargs)

    def available(self) -> list[str]:
        return sorted(self._factories)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.lower() in self._factories

    def __iter__(self) -> Iterator[str]:
        return iter(self.available())

    def __len__(self) -> int:
        return len(self._factories)
