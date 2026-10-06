"""Synthetic compact metadata, collision safety, spawn and single-read integrity checks."""
import hashlib
import json
import multiprocessing as mp
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from eksperimen_model.datasets.m4human_dataset import (
    ManifestSequence, ManifestView, UIDIndex, checked_tensor_load,
)
from eksperimen_model.m4human_training import WindowTensorCache
from eksperimen_model.utils.m4human_runtime import CONTRACT_VERSION, atomic_json, file_sha256


class CollisionIndex(UIDIndex):
    @staticmethod
    def hash_uid(uid):
        return 7


def raises(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('expected rejection')


def child_read(rows, index, queue):
    queue.put((rows[3]['window_id'], index['window3'], index.get('absent'),
               rows.offsets.is_shared(), index.hashes.is_shared(), index.indices.is_shared()))


def main():
    with tempfile.TemporaryDirectory(prefix='m4human_memory_', ignore_cleanup_errors=True) as directory:
        root = Path(directory)
        path = root/'index.jsonl'
        with path.open('w') as stream:
            for i in range(5000):
                stream.write(json.dumps({'window_id':f'window{i}', 'text':'provenance '*1000})+'\n')
        before = file_sha256(path)
        rows = ManifestSequence(path,identity_field='window_id')
        view = ManifestView(rows,[3,1,0])
        index = UIDIndex(rows,'window_id')
        assert [r['window_id'] for r in view] == ['window3','window1','window0']
        assert index['window3'] == 3 and index.get('missing') is None
        # ForkingPickler is the Windows worker transport; shared arrays carry handles.
        assert len(ForkingPickler.dumps((rows,view,index))) < 16000
        first_storage = rows.offsets.data_ptr()
        ForkingPickler.dumps(rows)
        assert rows.offsets.data_ptr() == first_storage
        ctx = mp.get_context('spawn')
        queue = ctx.Queue()
        process = ctx.Process(target=child_read,args=(rows,index,queue))
        process.start()
        result = queue.get(timeout=30)
        process.join(30)
        assert process.exitcode == 0 and result == ('window3',3,None,True,True,True)
        assert file_sha256(path) == before
        collisions = CollisionIndex([{'window_id':'a'},{'window_id':'b'}],'window_id')
        assert collisions['a'] == 0 and collisions['b'] == 1 and collisions.get('c') is None
        raises(lambda:CollisionIndex([{'window_id':'a'},{'window_id':'a'}],'window_id'))
        raises(lambda:UIDIndex([{'window_id':'a'},{'window_id':'a'}],'window_id'))
        duplicate = root/'duplicate.jsonl'
        duplicate.write_text('{"qa_id":"a","question":"one"}\n{"qa_id":"a","question":"two"}\n')
        raises(lambda:ManifestSequence(duplicate,identity_field='qa_id',strict_json=True))
        duplicate.write_text('{"qa_id":"a","qa_id":"b"}\n')
        raises(lambda:ManifestSequence(duplicate,identity_field='qa_id',strict_json=True))
        duplicate.write_text('{"qa_id":"a","answer":NaN}\n')
        raises(lambda:ManifestSequence(duplicate,identity_field='qa_id',strict_json=True))
        cache_root = root/'cache'
        cache_root.mkdir()
        tensor_path = cache_root/'window.pt'
        torch.save({'U':torch.ones(16,256)},tensor_path)
        record = {'window_id':'w','subject_id':1,'recording_id':'r','split':'train',
                  'tensor_path':'window.pt','tensor_sha256':file_sha256(tensor_path)}
        (cache_root/'index.jsonl').write_text(json.dumps(record)+'\n')
        atomic_json(cache_root/'metadata.json',{'contract_version':CONTRACT_VERSION,'complete':True,'count':1,
                     'lineage':{},'index_sha256':file_sha256(cache_root/'index.jsonl')})
        cache = WindowTensorCache(cache_root)
        assert isinstance(cache.rows,ManifestSequence)
        reads = []
        original_open = Path.open
        def tracked_open(path,*args,**kwargs):
            if path.resolve() == tensor_path.resolve() and args and args[0] == 'rb':
                reads.append(path)
            return original_open(path,*args,**kwargs)
        with patch.object(Path,'open',tracked_open):
            assert torch.equal(cache[0]['sensor']['U'],torch.ones(16,256))
        assert len(reads) == 1
        raises(lambda:checked_tensor_load(tensor_path,record['tensor_sha256'],max_bytes=1))
        with tensor_path.open('ab') as stream:
            stream.write(b'changed')
        raises(lambda:cache[0])
    print('PASS compact lazy metadata; shared spawn transport <16KiB; exact collision lookup; source immutable; single-read checked tensors')


if __name__ == '__main__':
    main()
