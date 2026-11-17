"""Generate report tables/figure and an Overleaf source bundle from release artifacts."""
from pathlib import Path
import json
import shutil
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/overleaf'
NAMES = {'logreg':'Logistic Regression', 'xgboost':'XGBoost', 'mlp':'Compact MLP', 'deep_mlp':'Deep MLP', 'resnet_mlp':'Residual MLP'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((ROOT/'outputs/release/release_summary.json').read_text())
    budgets = summary['budgets']
    lines = []
    for group, title in [('all','All 28 features'), ('low_level','21 low-level features')]:
        lines += [r'\subsection*{' + title + r': test ROC-AUC}', r'\begin{center}\begin{tabular}{lrrr}\toprule', r'Model & 250k & 1M & 2.5M \\\midrule']
        for model, name in NAMES.items():
            values = [next(r['roc_auc'] for r in budgets[b]['results'] if r['model']==model and r['group']==group) for b in ['250k','1m','2.5m']]
            lines.append(name + ' & ' + ' & '.join(f'{v:.4f}' for v in values) + r' \\')
        lines += [r'\bottomrule\end{tabular}\end{center}']
    lines += [r'\subsection*{2.5M events, all features: trade-offs}', r'\begin{center}\small\begin{tabular}{lrrrrr}\toprule', r'Model & ROC-AUC & AP & Brier & ECE & Train (s) \\\midrule']
    for model, name in NAMES.items():
        r=next(r for r in budgets['2.5m']['results'] if r['model']==model and r['group']=='all')
        lines.append(name+' & '+' & '.join(f'{r[k]:.4f}' for k in ['roc_auc','pr_auc','brier','ece'])+f" & {r['train_seconds']:.1f}"+r' \\')
    lines += [r'\bottomrule\end{tabular}\end{center}']
    (OUT/'results.tex').write_text('\n'.join(lines)+'\n')
    fig, axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
    for ax, group in zip(axes,['all','low_level']):
        for model,name in NAMES.items():
            vals=[next(r['roc_auc'] for r in budgets[b]['results'] if r['model']==model and r['group']==group) for b in ['250k','1m','2.5m']]
            ax.plot([.25,1,2.5],vals,marker='o',label=name)
        ax.set(title='All features' if group=='all' else 'Low-level features',xlabel='Sample budget (millions)',ylabel='Test ROC-AUC')
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=7,loc='lower right')
    fig.savefig(OUT/'scaling.png',dpi=180)
    plt.close(fig)
    shutil.copyfile(ROOT/'reports/saved_prediction_audit.json',OUT/'saved_prediction_audit.json')
    (OUT/'README.md').write_text('# Overleaf source\n\nUpload HiggsLens_Overleaf.zip as a new Overleaf project. Set main.tex as the main document and use pdfLaTeX. References use BibTeX.\n\nLocally: `python scripts/build_research_report.py`, then `tectonic reports/overleaf/main.tex`. Run the generator again after compilation to refresh the source ZIP if needed.\n\nResults are recorded release artifacts, not new training. See the report for limitations.\n')
    with zipfile.ZipFile(OUT/'HiggsLens_Overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['main.tex','results.tex','references.bib','scaling.png','saved_prediction_audit.json','README.md']:
            z.write(OUT/name,name)
    print(OUT/'HiggsLens_Overleaf.zip')

if __name__=='__main__':
    main()
