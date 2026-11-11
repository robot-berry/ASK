import unittest

from test7_ocr.runtime_memory import should_release_cached_vram


GIB = 1024 * 1024 * 1024


class RuntimeMemoryTests(unittest.TestCase):
    def test_resident_service_reuses_cache_with_ample_headroom(self) -> None:
        self.assertFalse(
            should_release_cached_vram(
                total_bytes=8 * GIB,
                reserved_bytes=4 * GIB,
                free_bytes=3 * GIB,
            )
        )

    def test_high_reserved_fraction_uses_original_cleanup(self) -> None:
        self.assertTrue(
            should_release_cached_vram(
                total_bytes=8 * GIB,
                reserved_bytes=7 * GIB,
                free_bytes=2 * GIB,
            )
        )

    def test_low_free_memory_uses_original_cleanup(self) -> None:
        self.assertTrue(
            should_release_cached_vram(
                total_bytes=8 * GIB,
                reserved_bytes=4 * GIB,
                free_bytes=512 * 1024 * 1024,
            )
        )

    def test_unknown_memory_state_fails_safe(self) -> None:
        self.assertTrue(
            should_release_cached_vram(
                total_bytes=0,
                reserved_bytes=0,
                free_bytes=0,
            )
        )


if __name__ == "__main__":
    unittest.main()
