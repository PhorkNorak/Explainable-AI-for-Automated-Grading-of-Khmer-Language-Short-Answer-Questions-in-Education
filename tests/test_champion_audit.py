import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments import exp10_significance as exp


class ChampionAuditTests(unittest.TestCase):
    def test_missing_champion_does_not_replace_existing_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'champion_metrics.csv'
            path.write_text('original evidence\n')
            with patch.object(exp, 'CHAMPS', {'classical': str(Path(tmp)/'missing')}):
                with self.assertRaises((FileNotFoundError, FileExistsError)):
                    exp.champion_metrics(tmp)
            self.assertEqual(path.read_text(), 'original evidence\n')

if __name__ == '__main__':
    unittest.main()
