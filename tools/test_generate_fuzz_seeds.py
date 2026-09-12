import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import generate_fuzz_seeds as seeds


class GenerateFuzzSeedsTests(unittest.TestCase):
    def test_encode_config_packs_the_live_eight_byte_prologue(self):
        header = seeds.encode_config(
            sr_idx=3,
            channels=2,
            app_idx=1,
            bitrate=64_000,
            complexity=8,
            vbr=True,
            vbr_constraint=True,
            inband_fec=2,
            dtx=True,
            loss_perc=37,
        )

        self.assertEqual(len(header), 8)
        self.assertEqual(header[:3], bytes([3, 1, 1]))
        self.assertEqual(header[3:5], struct.pack("<H", 58_000))
        self.assertEqual(header[5], 8)
        self.assertEqual(header[6], 0b1011)
        self.assertEqual(header[7], 1 | (37 << 1))

    def test_encode_seeds_put_the_first_pcm_sample_after_the_prologue(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(seeds, "CORPUS_DIR", temp), patch.object(
                seeds, "ENCODE_CONFIGS", [(4, 1, 1, 64_000, 5)]
            ), patch.object(
                seeds,
                "PATTERN_GENERATORS",
                [("marker", lambda n, _sr: [1234] * n)],
            ):
                count = seeds.generate_encode_seeds()

            self.assertEqual(count, 2)
            for target in (
                "fuzz_encode",
                "fuzz_roundtrip",
            ):
                files = list((Path(temp) / target).glob("*.bin"))
                self.assertEqual(len(files), 1)
                data = files[0].read_bytes()
                self.assertEqual(len(data[:8]), 8)
                self.assertEqual(data[:8], seeds.encode_config(4, 1, 1, 64_000, 5))
                self.assertEqual(struct.unpack_from("<h", data, 8)[0], 1234)


if __name__ == "__main__":
    unittest.main()
