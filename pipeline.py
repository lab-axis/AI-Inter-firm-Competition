"""Explicit entry points; no network calls or training unless selected."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
B = ROOT / 'B-MTGNN'
sys.path.insert(0, str(B))
from run_profile import load_profile, save_training_profile
from split_policy import build as build_split
HORIZONS = (6, 12, 24, 36)
INPUT = ROOT / 'data_preparation/data/5_bmtgnn_input'


def run_name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value):
        raise argparse.ArgumentTypeError('Use letters, digits, underscores and hyphens only.')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Print command without running it')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check', help='Read-only inventory; does not contact services')
    data = sub.add_parser('data')
    data.add_argument('step', choices=['collect', 'topics', 'build', 'export'])
    rag = sub.add_parser('rag')
    rag.add_argument('step', choices=['sql', 'preprocess', 'build'])
    train = sub.add_parser('train', help='New fixed-HP training; never overwrite a supplied run')
    train.add_argument('--run', type=run_name, required=True)
    train.add_argument('--horizon', type=int, choices=HORIZONS, default=12)
    train.add_argument('--split-policy', choices=['withheld_matched', 'withheld', 'legacy'], default='withheld_matched')
    train.add_argument('--valid-span', type=int)
    train.add_argument('--seed', type=int, default=2000)
    train.add_argument('--repeats', type=int, default=1)
    train.add_argument('--device', default='cpu')
    forecast = sub.add_parser('forecast')
    selection = forecast.add_mutually_exclusive_group()
    selection.add_argument('--run', type=run_name)
    selection.add_argument('--horizon', type=int, choices=HORIZONS)
    forecast.add_argument('--num-runs', type=int, default=50)
    forecast.add_argument('--device', default='cpu')
    forecast.add_argument('--output-dir', type=Path)
    agents = sub.add_parser('agents')
    agents.add_argument('--companies', nargs='+', required=True)
    agents.add_argument('--months', nargs='+', required=True, help='YYYY-MM, within forecast file')
    selection = agents.add_mutually_exclusive_group()
    selection.add_argument('--forecast-dir', type=Path)
    selection.add_argument('--run', type=run_name)
    selection.add_argument('--horizon', type=int, choices=HORIZONS)
    args, extra = parser.parse_known_args()
    if extra and args.command not in ('data', 'rag'):
        parser.error(f'Unrecognised arguments: {extra}')
    if extra[:1] == ['--']:
        extra = extra[1:]

    if args.command == 'check':
        paths = ['data_preparation/data/5_bmtgnn_input/bmtgnn_data.npy',
                 'data_preparation/data/5_bmtgnn_input/node_ids.csv',
                 'Agent/Multi_agent_Debate/main.py']
        paths += [f'B-MTGNN/Bayesian/{h}mo/o_model.safetensors' for h in HORIZONS]
        modules = ['torch', 'numpy', 'pandas', 'scipy', 'matplotlib', 'yaml',
                   'safetensors', 'optuna', 'aiohttp', 'langgraph', 'pydantic',
                   'neo4j', 'dotenv', 'requests', 'trafilatura', 'bs4', 'newspaper',
                   'sec_api', 'sentence_transformers', 'yfinance', 'pytrends']
        print(json.dumps({'files': {p: (ROOT / p).is_file() for p in paths},
                          'modules': {m: importlib.util.find_spec(m) is not None for m in modules},
                          'external_services': 'NOT checked (Neo4j, embeddings, chat)'}, indent=2))
        return

    cwd = ROOT
    if args.command == 'data':
        scripts = dict(collect='1.fetch_sec_filings.py', topics='2.topic_modeling.py',
                       build='3.build_dataset.py', export='4.export_bmtgnn.py')
        cwd = ROOT / 'data_preparation'
        command = [sys.executable, str(cwd / scripts[args.step]), *extra]
    elif args.command == 'rag':
        if args.step == 'build':
            cwd = ROOT / 'Agent'
            command = [sys.executable, str(cwd / 'auto_rag_builder.py'), *extra]
        else:
            command = [sys.executable, str(ROOT / 'Agent/TemporalRAG/engine/gdelt_collector.py'),
                       '--mode', args.step, *extra]
    elif args.command == 'train':
        if args.repeats < 1:
            parser.error('--repeats must be positive')
        destination = B / 'Bayesian' / args.run
        if destination.exists():
            parser.error('Run directory exists. Choose a NEW --run to preserve existing weights.')
        preset = load_profile(B / f'Bayesian/{args.horizon}mo', INPUT / 'bmtgnn_data.npy', INPUT / 'node_ids.csv')
        valid_span = args.valid_span if args.valid_span is not None else max(24, args.horizon)
        test_reserve = max(24, args.horizon)
        # Check feasibility before creating files or starting training. Do not silently
        # substitute a historical split when a target-disjoint partition does not fit.
        try:
            build_split(144, args.horizon, 1, args.horizon, policy=args.split_policy,
                        test_reserve=test_reserve, valid_span=valid_span)
        except ValueError as error:
            parser.error(f'Training split is not feasible: {error}')
        if not args.dry_run:
            destination.mkdir(parents=True)
            (destination / 'hp.txt').write_text(repr(preset['training_preset_hps']), encoding='utf-8')
        cwd = B
        command = [sys.executable, str(B / 'train_test.py'), '--version', args.run,
                   '--months', str(args.horizon), '--fixed_hp', 'True', '--seed', str(args.seed),
                   '--search_iters', str(args.repeats), '--device', args.device,
                   '--split_policy', args.split_policy, '--test_reserve', str(test_reserve), '--valid_span', str(valid_span)]
    elif args.command == 'forecast':
        cwd = B
        selected_run = args.run or f'{args.horizon or 12}mo'
        command = [sys.executable, str(B / 'forecast.py'), '--run', selected_run,
                   '--num_runs', str(args.num_runs), '--device', args.device]
        if args.output_dir:
            command += ['--output-dir', str(args.output_dir.resolve())]
    else:
        cwd = ROOT / 'Agent'
        selected_run = args.run or f'{args.horizon or 12}mo'
        forecast_dir = args.forecast_dir or B / 'Bayesian' / selected_run / 'forecast/data'
        command = [sys.executable, str(cwd / 'Multi_agent_Debate/main.py'),
                   '--companies', *args.companies, '--months', *args.months,
                   '--forecast-dir', str(forecast_dir.resolve())]
    print('cwd:', cwd, flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=cwd, check=True)
        if args.command == 'train':
            save_training_profile(destination, INPUT / 'bmtgnn_data.npy', INPUT / 'node_ids.csv',
                                  args.split_policy, test_reserve, valid_span)


if __name__ == '__main__':
    main()
