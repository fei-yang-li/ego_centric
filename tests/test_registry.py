from __future__ import annotations

import unittest

from ego_vla.registry import Registry


class RegistryTests(unittest.TestCase):
    def test_register_and_create(self) -> None:
        registry: Registry[str] = Registry("widget")

        @registry.register("foo")
        def _build_foo() -> str:
            return "foo-instance"

        registry.register("bar", lambda: "bar-instance")

        self.assertEqual(registry.create("foo"), "foo-instance")
        self.assertEqual(registry.create("bar"), "bar-instance")
        self.assertEqual(registry.available(), ["bar", "foo"])
        self.assertIn("foo", registry)
        self.assertEqual(len(registry), 2)

    def test_create_is_case_insensitive_and_defaults_to_none(self) -> None:
        registry: Registry[str] = Registry("widget")
        registry.register("none", lambda: "noop")
        self.assertEqual(registry.create("NONE"), "noop")
        self.assertEqual(registry.create(None), "noop")

    def test_duplicate_registration_fails(self) -> None:
        registry: Registry[str] = Registry("widget")
        registry.register("foo", lambda: "a")
        with self.assertRaises(ValueError):
            registry.register("foo", lambda: "b")
        registry.register("foo", lambda: "c", override=True)
        self.assertEqual(registry.create("foo"), "c")

    def test_unknown_backend_lists_available(self) -> None:
        registry: Registry[str] = Registry("widget")
        registry.register("foo", lambda: "a")
        with self.assertRaises(ValueError) as ctx:
            registry.create("missing")
        self.assertIn("foo", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
