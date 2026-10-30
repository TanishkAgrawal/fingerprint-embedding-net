import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pandas as pd
from src.retrieval.subject_comparison import compare_subject

class SubjectComparisonTests(unittest.TestCase):
    def setUp(self):
        self.meta = pd.DataFrame({'embedding_row': [0,1,2,3,4],
            'subject_id': [102,102,102,538,102],
            'class_id': ['102_Left_middle','102_Left_middle','102_Left_index','538_Left_ring','102_Left_middle'],
            'filepath': ['query.bmp','other.bmp','index.bmp','foreign.bmp','query.bmp']})
        self.remote = Mock()
        self.vectors = np.eye(5, dtype=np.float32)

    def test_excludes_self_and_duplicate_image_but_keeps_other_capture(self):
        self.remote.query.return_value = SimpleNamespace(matches=[
            SimpleNamespace(id=str(i),score=s) for i,s in [(0,1),(4,1),(1,.92),(2,.26)]])
        result = compare_subject(self.remote,'archive',self.meta,self.vectors,0)
        self.assertEqual([r['Image ID'] for r in result],[1,2])
        self.assertEqual([r['Same finger'] for r in result],[True,False])
        self.assertEqual([r['Rank'] for r in result],[1,2])
        self.assertEqual(self.remote.query.call_args.kwargs['filter'],
            {'class_id': {'$in': ['102_Left_middle','102_Left_index']}})

    def test_foreign_subject_is_rejected(self):
        self.remote.query.return_value = SimpleNamespace(matches=[SimpleNamespace(id='3',score=.8)])
        with self.assertRaisesRegex(ValueError,'different subject'):
            compare_subject(self.remote,'archive',self.meta,self.vectors,0)

    def test_top_k_applies_after_self_exclusion(self):
        self.remote.query.return_value = SimpleNamespace(matches=[
            SimpleNamespace(id=str(i),score=.9) for i in (0,1,2)])
        self.assertEqual(len(compare_subject(self.remote,'archive',self.meta,self.vectors,0,1)),1)

if __name__ == '__main__':
    unittest.main()
