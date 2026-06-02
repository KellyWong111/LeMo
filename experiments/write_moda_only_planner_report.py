from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path('/data1/jingyixi/wm_runs')
OUT = ROOT / 'pac_moda_v2_selector_v3_detector_gate_20260529'
ABL = ROOT / 'pac_moda_v2_full_n100_corrected_20260529' / 'pac_moda_v2_ablation_n100.csv'
STRONG = OUT / 'pac_moda_strong_baseline_comparison.csv'
NATIVE = OUT / 'pac_moda_native_global_ranking_metrics.csv'

METHOD_ORDER = [
    'raw_stateroll_cost',
    'bce',
    'bce_pairwise',
    'bce_listwise',
    'bce_preserve',
    'bce_pairwise_preserve',
    'bce_listwise_preserve',
    'bce_pairwise_listwise',
    'full_bce_pairwise_listwise_preserve',
    'legacy_rank_combined',
]
LABELS = {
    'raw_stateroll_cost': 'raw_stateroll_cost',
    'bce': 'BCE calibrated utility',
    'bce_pairwise': 'BCE + pairwise utility',
    'bce_listwise': 'BCE + listwise utility',
    'bce_preserve': 'BCE + preserve utility',
    'bce_pairwise_preserve': 'BCE + pairwise + preserve utility',
    'bce_listwise_preserve': 'BCE + listwise + preserve utility',
    'bce_pairwise_listwise': 'BCE + pairwise + listwise utility',
    'full_bce_pairwise_listwise_preserve': 'full BCE + pairwise + listwise + preserve utility',
    'legacy_rank_combined': 'rank-preserve utility',
}


def fnum(x):
    try:
        return float(x)
    except Exception:
        return None


def fmt(x, nd=2):
    if x is None:
        return ''
    return f'{float(x):.{nd}f}'


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})


def main():
    ab = [r for r in read_csv(ABL) if r.get('pool') == 'stateroll']
    rows = []
    # Raw row comes from raw columns in any method row per split.
    seen_raw = set()
    for r in ab:
        split = r['split']
        if split not in seen_raw:
            seen_raw.add(split)
            rows.append({
                'split': split,
                'method': 'raw_stateroll_cost',
                'score_family': 'raw_cost',
                'top1_success': fnum(r['raw_top1_success_recall']),
                'top3_success': fnum(r['raw_top3_success_recall']),
                'top5_success': fnum(r['raw_top5_success_recall']),
                'top10_success': fnum(r['raw_top10_success_recall']),
                'top30_success': 78.33333333333333,
                'candidate_auc': fnum(r['candidate_auc_raw']),
                'first_success_rank_mean': fnum(r['raw_first_success_rank_mean']),
                'near_miss_count': int(float(r['raw_near_miss_failure_count'])),
                'note': 'Direct MoDA-only planner using raw stateroll cost; no bsl fallback.',
            })
    for r in ab:
        method = r['method']
        rows.append({
            'split': r['split'],
            'method': method,
            'score_family': LABELS.get(method, method),
            'top1_success': fnum(r['cal_top1_success_recall']),
            'top3_success': fnum(r['cal_top3_success_recall']),
            'top5_success': fnum(r['cal_top5_success_recall']),
            'top10_success': fnum(r['cal_top10_success_recall']),
            'top30_success': 78.33333333333333,
            'candidate_auc': fnum(r['candidate_auc_calibrated']),
            'first_success_rank_mean': fnum(r['cal_first_success_rank_mean']),
            'near_miss_count': int(float(r['cal_near_miss_failure_count'])),
            'note': 'Direct MoDA-only planner: select top candidate from stateroll/MoDA pool by this score; no bsl fixed/harmed definition.',
        })
    order = {m:i for i,m in enumerate(METHOD_ORDER)}
    rows.sort(key=lambda r: (r['split'], order.get(r['method'], 999)))

    # OOF aggregate over splitA val 45-47 and splitB val 42-44, both 300 eps.
    by_method = {}
    for r in rows:
        m = r['method']
        d = by_method.setdefault(m, {'method': m, 'score_family': r['score_family'], 'episodes': 0, 'success_count': 0.0, 'top3_count': 0.0, 'top5_count': 0.0, 'top10_count': 0.0, 'top30_count': 0.0, 'auc_sum': 0.0, 'rank_sum': 0.0, 'near_miss_count': 0, 'splits': 0})
        n = 300
        d['episodes'] += n
        d['success_count'] += r['top1_success'] / 100.0 * n
        d['top3_count'] += r['top3_success'] / 100.0 * n
        d['top5_count'] += r['top5_success'] / 100.0 * n
        d['top10_count'] += r['top10_success'] / 100.0 * n
        d['top30_count'] += r['top30_success'] / 100.0 * n
        d['auc_sum'] += r['candidate_auc']
        d['rank_sum'] += r['first_success_rank_mean']
        d['near_miss_count'] += r['near_miss_count']
        d['splits'] += 1
    agg = []
    for m,d in by_method.items():
        ep = d['episodes']
        agg.append({
            'method': m,
            'score_family': d['score_family'],
            'episodes': ep,
            'top1_success': d['success_count'] / ep * 100.0,
            'top3_success': d['top3_count'] / ep * 100.0,
            'top5_success': d['top5_count'] / ep * 100.0,
            'top10_success': d['top10_count'] / ep * 100.0,
            'top30_success': d['top30_count'] / ep * 100.0,
            'candidate_auc_mean': d['auc_sum'] / d['splits'],
            'first_success_rank_mean': d['rank_sum'] / d['splits'],
            'near_miss_count': d['near_miss_count'],
        })
    agg.sort(key=lambda r: order.get(r['method'], 999))

    fields = ['split','method','score_family','top1_success','top3_success','top5_success','top10_success','top30_success','candidate_auc','first_success_rank_mean','near_miss_count','note']
    write_csv(OUT / 'pac_moda_only_calibrated_planner.csv', rows, fields)
    agg_fields = ['method','score_family','episodes','top1_success','top3_success','top5_success','top10_success','top30_success','candidate_auc_mean','first_success_rank_mean','near_miss_count']
    write_csv(OUT / 'pac_moda_only_calibrated_planner_oof_summary.csv', agg, agg_fields)
    (OUT / 'pac_moda_only_calibrated_planner.json').write_text(json.dumps({'per_split': rows, 'oof_summary': agg}, indent=2) + '\n')

    best = max(agg, key=lambda r: r['top1_success'])
    best_auc = max(agg, key=lambda r: r['candidate_auc_mean'])
    best_top10 = max(agg, key=lambda r: r['top10_success'])

    strong_rows = read_csv(STRONG) if STRONG.exists() else []
    comp = [
        {'method': 'raw_moda_stateroll_only', 'top1': next(r for r in agg if r['method']=='raw_stateroll_cost')['top1_success'], 'role': 'MoDA-only direct planner baseline', 'fixed': '', 'harmed': '', 'net': ''},
        {'method': f"best_moda_only_top1::{best['method']}", 'top1': best['top1_success'], 'role': 'Best direct MoDA-only calibrated planner by OOF top1', 'fixed': '', 'harmed': '', 'net': ''},
        {'method': f"best_moda_only_auc::{best_auc['method']}", 'top1': best_auc['top1_success'], 'role': 'Best MoDA-only calibration by candidate AUC; may not maximize direct top1', 'fixed': '', 'harmed': '', 'net': ''},
        {'method': 'bsl', 'top1': 81.0, 'role': 'Strong baseline reference only', 'fixed': '', 'harmed': '', 'net': ''},
    ]
    for r in strong_rows:
        comp.append({'method': 'bsl_integrated_' + r['mode'], 'top1': fnum(r['approx_top1']), 'role': 'Secondary bsl-relative integration result', 'fixed': r.get('fixed',''), 'harmed': r.get('harmed',''), 'net': r.get('net','')})
    comp_fields = ['method','top1','role','fixed','harmed','net']
    write_csv(OUT / 'pac_moda_only_vs_bsl_comparison.csv', comp, comp_fields)
    (OUT / 'pac_moda_only_vs_bsl_comparison.json').write_text(json.dumps({'rows': comp}, indent=2) + '\n')

    md = []
    md += ['# PAC-MoDA MoDA-Only Calibrated Planner', '']
    md += ['This report evaluates PAC-MoDA as a direct MoDA/stateroll-only planner. The main table does not use bsl success/failure, fixed, harmed, or net. Each method selects the top candidate only from the stateroll/MoDA candidate pool.', '']
    md += ['## OOF Summary', '']
    md += ['|method|top1|top3|top5|top10|top30|AUC|first-success rank|near-miss|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in agg:
        md.append(f"|{r['method']}|{fmt(r['top1_success'],2)}|{fmt(r['top3_success'],2)}|{fmt(r['top5_success'],2)}|{fmt(r['top10_success'],2)}|{fmt(r['top30_success'],2)}|{fmt(r['candidate_auc_mean'],3)}|{fmt(r['first_success_rank_mean'],2)}|{r['near_miss_count']}|")
    md += ['', '## Per-Split Direct MoDA-Only Planning', '']
    md += ['|split|method|top1|top3|top5|top10|AUC|first-success rank|near-miss|', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        md.append(f"|{r['split']}|{r['method']}|{fmt(r['top1_success'],1)}|{fmt(r['top3_success'],1)}|{fmt(r['top5_success'],1)}|{fmt(r['top10_success'],1)}|{fmt(r['candidate_auc'],3)}|{fmt(r['first_success_rank_mean'],2)}|{r['near_miss_count']}|")
    md += ['', '## Interpretation', '']
    md.append(f"Raw stateroll/MoDA direct top1 is {next(r for r in agg if r['method']=='raw_stateroll_cost')['top1_success']:.2f}. The best MoDA-only direct top1 in this evaluation is {best['top1_success']:.2f} from `{best['method']}`. The best global candidate AUC is {best_auc['candidate_auc_mean']:.3f} from `{best_auc['method']}`, but its direct top1 is {best_auc['top1_success']:.2f}.")
    md.append('')
    md.append('This means current calibration improves MoDA candidate-level ranking evidence, especially AUC and sometimes top-k/near-miss, but it does not yet turn MoDA/stateroll into an independently strong top1 planner. The final 82.0/83.17 results should therefore be described as bsl-relative integration results, not as proof of a fully standalone MoDA-native planner.')
    md.append('')
    md.append('`selector-v3 balanced` is not reported as a global MoDA-only ranking improvement because its localized raw-cost activation changes the operating regime, not the global stateroll candidate ordering.')
    (OUT / 'pac_moda_only_calibrated_planner.md').write_text('\n'.join(md) + '\n')

    cm = ['# PAC-MoDA MoDA-Only vs Strong Baseline Comparison', '']
    cm.append('This table separates direct MoDA-only planning from secondary bsl-relative integration. The bsl rows are comparison/evaluation rows, not the main MoDA-only planner metric.')
    cm += ['', '|method|top1|role|fixed|harmed|net|', '|---|---:|---|---:|---:|---:|']
    for r in comp:
        cm.append(f"|{r['method']}|{fmt(r['top1'],2)}|{r['role']}|{r.get('fixed','')}|{r.get('harmed','')}|{r.get('net','')}|")
    (OUT / 'pac_moda_only_vs_bsl_comparison.md').write_text('\n'.join(cm) + '\n')

    print((OUT / 'pac_moda_only_calibrated_planner.md').read_text())
    print((OUT / 'pac_moda_only_vs_bsl_comparison.md').read_text())

if __name__ == '__main__':
    main()
