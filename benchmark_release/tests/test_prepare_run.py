import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('prepare_run', Path(__file__).resolve().parents[1] / 'prepare_run.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)

class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name); self.old = (p.ROOT, p.BUNDLE)
        self.addCleanup(self.restore)
        p.ROOT = root / 'repo'; p.BUNDLE = p.ROOT / 'benchmark_release'
        p.BUNDLE.mkdir(parents=True); self.data = root / 'data'; self.data.mkdir(); self.out = root / 'output'
        (p.ROOT / 'model.py').write_text('original source')
        (self.data / 'data.npz').write_bytes(b'corrected data'); (self.data / 'graph.npz').write_bytes(b'graph')
        self.protocol = {'source_sha256': {'model.py': p.sha(p.ROOT / 'model.py')}}
        self.save_protocol()
        self.manifest = {'frozen_protocol_path': 'benchmark_release/protocol.json', 'frozen_protocol_sha256': self.digest(), 'years': [{'data': 'data.npz', 'graph': 'graph.npz', 'data_sha256': p.sha(self.data / 'data.npz'), 'graph_sha256': p.sha(self.data / 'graph.npz')}]}
        self.save_manifest()
        (p.BUNDLE / 'index.json').write_text(json.dumps([{'label':'test', 'district':'PEMS04', 'manifest':'manifest.json', 'original_manifest_sha256':'test-original', 'portable_manifest_sha256':'test-portable'}]))
    def restore(self):p.ROOT, p.BUNDLE = self.old
    def digest(self):return p.hashlib.sha256(json.dumps(self.protocol, sort_keys=True).encode()).hexdigest()
    def save_protocol(self):(p.BUNDLE / 'protocol.json').write_text(json.dumps(self.protocol))
    def save_manifest(self):(p.BUNDLE / 'manifest.json').write_text(json.dumps(self.manifest))
    def test_valid_preparation_does_not_launch_or_copy_data(self):
        p.prepare('test', self.data, self.out)
        self.assertTrue((self.out / 'model.py').is_file()); self.assertFalse((self.out / 'data.npz').exists())
        self.assertEqual(json.loads((self.out / 'manifest.json').read_text())['frozen_protocol_sha256'], self.digest())
    def test_corrupt_data_rejected_before_creating_output(self):
        (self.data / 'data.npz').write_bytes(b'wrong')
        with self.assertRaises(ValueError):p.prepare('test', self.data, self.out)
        self.assertFalse(self.out.exists())
    def test_original_overlay_selects_frozen_source(self):
        overlay=p.BUNDLE / 'pems04_overlay/model.py'; overlay.parent.mkdir(); overlay.write_text('original source')
        (p.ROOT / 'model.py').write_text('newer source')
        p.prepare('test',self.data,self.out)
        self.assertEqual((self.out/'model.py').read_text(),'original source')
    def test_path_escape_rejected(self):
        self.manifest['years'][0]['data']='../outside'; self.save_manifest()
        with self.assertRaises(ValueError):p.prepare('test',self.data,self.out)
        self.assertFalse(self.out.exists())
    def test_existing_output_preserved(self):
        self.out.mkdir();(self.out/'sentinel').write_text('keep')
        with self.assertRaises(FileExistsError):p.prepare('test',self.data,self.out)
        self.assertEqual((self.out/'sentinel').read_text(),'keep')

if __name__=='__main__':unittest.main()
