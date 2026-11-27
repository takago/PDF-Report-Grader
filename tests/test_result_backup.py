import tempfile
import unittest
from pathlib import Path
from report_grader.collector import backup_existing_output, prepare_reports

class ResultBackupTests(unittest.TestCase):
    def test_existing_folder_without_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'reports'
            root.mkdir()
            self.assertEqual(prepare_reports(root), root.resolve())
    def test_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'reports.xlsx'
            self.assertIsNone(backup_existing_output(output))
            output.write_bytes(b'original')
            first = backup_existing_output(output)
            output.write_bytes(b'updated')
            second = backup_existing_output(output)
            self.assertEqual(first.read_bytes(), b'original')
            self.assertEqual(second.read_bytes(), b'updated')
            self.assertNotEqual(first, second)
            self.assertEqual(output.read_bytes(), b'updated')
