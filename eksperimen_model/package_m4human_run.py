"""Validate and package a complete run without copying datasets or raw targets."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import zipfile
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eksperimen_model.utils.m4human_runtime import file_sha256, validate_run_artifacts

RUN_FILES = {'config_snapshot.yaml', 'lineage.json', 'history.jsonl', 'events.jsonl',
             'predictions_val.jsonl', 'predictions_train_panel.jsonl', 'diagnostic_samples.npz',
             'diagnostic_samples.json', 'metrics.json', 'metric_recomputation.json',
             'best.pt', 'last.pt', 'completion.json', 'report.md', 'loss_curve.png'}


def package_run(run_dir, output, audit_metadata=()):
    root, output = Path(run_dir), Path(output)
    issues = validate_run_artifacts(root)
    if issues:
        raise ValueError('run is incomplete or invalid: ' + ', '.join(issues))
    if output.exists():
        raise FileExistsError(output)
    files = {name: root / name for name in sorted(RUN_FILES) if (root / name).is_file()}
    # A QA witness must travel with its frozen reference, not an external path.
    witness_path = root / 'metric_recomputation.json'
    witness = json.loads(witness_path.read_text(encoding='utf-8'))
    if witness.get('kind') == 'qa':
        reference = Path(witness['qa_reference_path'])
        if reference.is_absolute() or '..' in reference.parts:
            raise ValueError('QA packaging requires a run-local relative reference')
        files[reference.as_posix()] = root / reference
    for path in audit_metadata:
        path = Path(path)
        if path.suffix.lower() not in ('.json', '.yaml', '.yml', '.md') or path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError('optional audit inputs must be small text metadata')
        name = 'audit_metadata/' + path.name
        if name in files:
            raise ValueError('duplicate audit metadata basename')
        files[name] = path
    checksums = {name: file_sha256(path) for name, path in files.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.zip', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, path in files.items():
                archive.write(path, name)
            archive.writestr('checksums.json', json.dumps(checksums, sort_keys=True, indent=2) + '\n')
        # Revalidate archive contents and exact bytes before atomic publication.
        with tempfile.TemporaryDirectory() as directory:
            with zipfile.ZipFile(temporary) as archive:
                archive.extractall(directory)
            for name, expected in checksums.items():
                if file_sha256(Path(directory) / name) != expected:
                    raise ValueError('archive checksum mismatch: ' + name)
            issues = validate_run_artifacts(directory)
            if issues:
                raise ValueError('archive validation failed: ' + ', '.join(issues))
        import os
        os.link(temporary, output)
        return {'archive': str(output), 'sha256': file_sha256(output), 'files': len(files),
                'artifact_integrity': 'verified', 'scientific_validity': 'requires_separate_gate_validation'}
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--audit-metadata', action='append', default=[])
    args = parser.parse_args()
    print(json.dumps(package_run(args.run_dir, args.output, args.audit_metadata), indent=2))

if __name__ == '__main__':
    main()
