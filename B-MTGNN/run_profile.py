"""Checkpoint architecture and input-scaling profiles for named runs."""
import hashlib
import json
from pathlib import Path
import struct


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_arch(path):
    with Path(path).open('rb') as handle:
        size = struct.unpack('<Q', handle.read(8))[0]
        if size > 16 * 1024 * 1024:
            raise ValueError('Unexpected checkpoint header size')
        header = json.loads(handle.read(size))
    return json.loads(header['__metadata__']['arch'])


def load_profile(run_dir, data_file, nodes_file):
    run_dir = Path(run_dir)
    profile = json.loads((run_dir / 'provenance.json').read_text(encoding='utf-8'))
    for path, key in ((run_dir / 'o_model.safetensors', 'checkpoint_sha256'),
                      (data_file, 'input_sha256'), (nodes_file, 'node_ids_sha256')):
        if file_hash(path) != profile[key]:
            raise ValueError(f'Artifact differs from run profile: {path}')
    if profile['architecture'] != checkpoint_arch(run_dir / 'o_model.safetensors'):
        raise ValueError('Run profile and checkpoint architecture differ')
    if profile['split_policy'] not in ('legacy', 'withheld', 'withheld_matched'):
        raise ValueError('Unsupported run-profile split policy')
    return profile


def save_training_profile(run_dir, data_file, nodes_file, split_policy, test_reserve, valid_span):
    """Record settings after a successful training subprocess; do not rewrite weights."""
    run_dir = Path(run_dir)
    profile = {
        'source_run': run_dir.name,
        'checkpoint_sha256': file_hash(run_dir / 'o_model.safetensors'),
        'input_sha256': file_hash(data_file), 'node_ids_sha256': file_hash(nodes_file),
        'architecture': checkpoint_arch(run_dir / 'o_model.safetensors'),
        'split_policy': split_policy, 'test_reserve': test_reserve, 'valid_span': valid_span,
        'normalize': 2, 'num_eval': 7, 'horizon_offset': 1,
        'panel_shape': [30, 144, 17], 'panel_start': '2014-01', 'panel_end': '2025-12',
        'profile_basis': 'Settings supplied by pipeline.py for this training run.',
        'external_adjacency': False,
    }
    (run_dir / 'provenance.json').write_text(json.dumps(profile, indent=2), encoding='utf-8')
