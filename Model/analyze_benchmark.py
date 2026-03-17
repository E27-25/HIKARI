import json, glob, os, re
from collections import defaultdict

RESULTS_DIR = './disease_classification_results'
pred_files = sorted(glob.glob(f'{RESULTS_DIR}/*_predictions.json'))

def parse_name(fname):
    base = os.path.basename(fname)
    m = re.match(r'results_disease_(.+?)_RAG(R\d)_(P\d)_val_predictions', base)
    if m:
        return m.group(1), m.group(2), m.group(3)
    m = re.match(r'results_disease_(.+?)_RAG(R\d)_val_predictions', base)
    if m:
        return m.group(1), m.group(2), 'P0'
    return None, None, None

DISEASES = [
    'squamous cell carcinoma in situ',
    'basal cell carcinoma',
    'melanocytic nevi',
    'psoriasis',
    'lupus erythematosus',
    'lichen planus',
    'scleroderma',
    'photodermatoses',
    'sarcoidosis',
    'acne vulgaris',
]

print(f'Total prediction files found: {len(pred_files)}')
print()

data = {}
for f in pred_files:
    model, rag, prompt = parse_name(f)
    if model is None:
        print(f'SKIP (no pattern match): {os.path.basename(f)}')
        continue
    with open(f, encoding='utf-8') as fp:
        raw = json.load(fp)

    # Handle both wrapped {"predictions": [...]} and bare list formats
    if isinstance(raw, dict) and 'predictions' in raw:
        preds = raw['predictions']
    elif isinstance(raw, list):
        preds = raw
    else:
        print(f'SKIP (unknown format): {os.path.basename(f)}')
        continue

    total = len(preds)
    error = sum(1 for p in preds if p.get('status') == 'error')
    # "success" status: model returned a label — check if it matches ground truth
    correct = sum(1 for p in preds
                  if p.get('status') == 'success' and p.get('predicted') == p.get('ground_truth'))
    wrong = sum(1 for p in preds
                if p.get('status') == 'success' and p.get('predicted') != p.get('ground_truth')
                and p.get('predicted') not in (None, 'Unknown', ''))
    unknown = sum(1 for p in preds
                  if p.get('status') == 'success' and p.get('predicted') in (None, 'Unknown', ''))

    disease_stats = defaultdict(lambda: {'correct': 0, 'wrong': 0, 'error': 0, 'unknown': 0, 'total': 0})
    for p in preds:
        gt = p.get('ground_truth', '')
        st = p.get('status', '')
        pred = p.get('predicted', '')
        disease_stats[gt]['total'] += 1
        if st == 'error':
            disease_stats[gt]['error'] += 1
        elif st == 'success':
            if pred == gt:
                disease_stats[gt]['correct'] += 1
            elif pred in (None, 'Unknown', ''):
                disease_stats[gt]['unknown'] += 1
            else:
                disease_stats[gt]['wrong'] += 1

    key = (model, rag, prompt)
    data[key] = {
        'total': total,
        'correct': correct,
        'wrong': wrong,
        'error': error,
        'unknown': unknown,
        'acc_valid': correct / (total - error - unknown) if (total - error - unknown) > 0 else 0,
        'acc_total': correct / 99,  # Always penalize using denominator of 99
        'disease_stats': dict(disease_stats),
    }

# --- Summary Table ---
models = ['fuzzytopk', 'fuzzytopk_s1cascade', 'M1']
rags = ['R0', 'R1', 'R2', 'R3', 'R4']
prompts = ['P0', 'P1', 'P2', 'P3']

print('=' * 110)
print('COVERAGE TABLE  |  Acc = correct/99 (penalizes errors/unknowns)  |  format: acc% (valid/99)')
print('=' * 110)

for model in models:
    print(f'\n--- Model: {model} ---')
    header = f'{"":6} | ' + ' | '.join(f'{p:22}' for p in prompts)
    print(header)
    print('-' * len(header))
    for rag in rags:
        row_parts = []
        for prompt in prompts:
            key = (model, rag, prompt)
            if key in data:
                d = data[key]
                acc = d['acc_total'] * 100
                valid = d['total'] - d['error'] - d['unknown']
                cell = f'{acc:5.1f}% ({valid:2d}/99)'
            else:
                cell = 'MISSING'
            row_parts.append(f'{cell:22}')
        print(f'{rag:6} | ' + ' | '.join(row_parts))

# --- Per-disease breakdown ---
print()
print('=' * 110)
print('PER-DISEASE BREAKDOWN  |  correct | wrong | unknown | error  |  bar: #=correct o=wrong ?=unknown E=error')
print('=' * 110)

for model in models:
    print(f'\n=== Model: {model} ===')
    for rag in rags:
        for prompt in prompts:
            key = (model, rag, prompt)
            if key not in data:
                continue
            d = data[key]
            valid = d['total'] - d['error'] - d['unknown']
            print(f'\n  {model} {rag} {prompt}  |  acc={d["acc_total"]*100:.1f}% ({d["correct"]}/99)'
                  f'  |  valid={valid}/99  |  error={d["error"]}  unk={d["unknown"]}')
            print(f'  {"Disease":<40} {"GT":>3} | {"OK":>3} | {"Wrong":>5} | {"Unk":>3} | {"Err":>3}')
            print(f'  {"-" * 72}')
            for disease in DISEASES:
                ds = d['disease_stats'].get(
                    disease, {'total': 0, 'correct': 0, 'wrong': 0, 'error': 0, 'unknown': 0})
                total_d = ds['total']
                if total_d == 0:
                    continue
                ok = ds['correct']
                wr = ds['wrong']
                unk = ds['unknown']
                err = ds['error']
                bar = '#' * ok + 'o' * wr + '?' * unk + 'E' * err
                print(f'  {disease:<40} {total_d:>3} | {ok:>3} | {wr:>5} | {unk:>3} | {err:>3}  {bar}')
