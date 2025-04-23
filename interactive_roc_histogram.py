import pandas as pd
import numpy as np
import math
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

from bokeh.layouts import row, column
from bokeh.models import (
    ColumnDataSource, Span, Div, CustomJS, Whisker, HoverTool, Title, Range1d
)
from bokeh.plotting import figure, curdoc
import panel as pn
from panel.widgets import FloatSlider
import threading
import os
from pathlib import Path

# Initialize Panel with Bokeh extension
pn.extension()

# Launch Panel server: panel serve interactive_roc_histogram.py   --address 172.26.113.23   --port 5006   --allow-websocket-origin=172.26.113.23:5006

# 1) At the top‐level, capture the Bokeh document
doc = pn.state.curdoc or curdoc()

# ─── Setup ─────────────────────────────────────────────────────────────────
# Determine script path (works in notebook or script)
try:
    script_path = Path(__file__).parent.resolve()
except NameError:
    script_path = Path().resolve()

data_dir = script_path / "data"

if not data_dir.is_dir():
    raise FileNotFoundError(f"Data directory not found: {data_dir}")

def find_input_files(data_dir: Path):
    """
    Walks data_dir/<feature_set>/<model>/<target>/<sample>/
    and collects all immediate subdirectories (ds) under each sample.

    Returns a nested dict:
      structure[feature_set][model][target][sample] = [ds1, ds2, …]
    """
    structure: dict = {}
    for fs in os.listdir(data_dir):
        fs_dir = data_dir / fs
        if not fs_dir.is_dir():
            continue
        structure[fs] = {}

        # model layer
        for model in os.listdir(fs_dir):
            model_dir = fs_dir / model
            if not model_dir.is_dir():
                continue
            structure[fs][model] = {}

            # target layer
            for target in os.listdir(model_dir):
                target_dir = model_dir / target
                if not target_dir.is_dir():
                    continue
                structure[fs][model][target] = {}

                # sample layer (new)
                for sample in os.listdir(target_dir):
                    sample_dir = target_dir / sample
                    if not sample_dir.is_dir():
                        continue

                    # the final ds directories under each sample
                    ds_list = [
                        d for d in os.listdir(sample_dir)
                        if (sample_dir / d).is_dir()
                    ]
                    if ds_list:
                        structure[fs][model][target][sample] = ds_list

    return structure

structure = find_input_files(data_dir)

# ─── Widgets ────────────────────────────────────────────────────────────────
shared_width = 175
feature_select = pn.widgets.Select(
    name="Feature Set", options=list(structure.keys()),
    value=list(structure.keys())[0], width=shared_width
)
model_select = pn.widgets.Select(
    name="Trained Model",
    options=list(structure[feature_select.value].keys()),
    value=list(structure[feature_select.value].keys())[0], width=shared_width
)
target_select = pn.widgets.Select(
    name="Target Dataset",
    options=list(structure[feature_select.value][model_select.value].keys()),
    value=list(structure[feature_select.value][model_select.value].keys())[0], width=shared_width
)
ground_truth_select = pn.widgets.Select(
    name="Ground Truth",
    options=list(structure[feature_select.value][model_select.value][target_select.value].keys()),
    value=list(structure[feature_select.value][model_select.value][target_select.value].keys())[0], width=shared_width
)
sample_select = pn.widgets.Select(
    name="Sample Select",
    options=structure[feature_select.value][model_select.value][target_select.value][ground_truth_select.value],
    value=structure[feature_select.value][model_select.value][target_select.value][ground_truth_select.value][0], width=shared_width
)

# cascade updates
@pn.depends(feature_select.param.value, watch=True)
def _update_models(fs):
    mlist = list(structure[fs].keys())
    model_select.options = mlist
    model_select.value = mlist[0]

@pn.depends(model_select.param.value, watch=True)
def _update_targets(model):
    fs = feature_select.value
    tlist = list(structure[fs][model].keys())
    target_select.options = tlist
    target_select.value = tlist[0]

@pn.depends(target_select.param.value, watch=True)
def _update_ground_truth(event):
    f = feature_select.value
    m = model_select.value
    t = target_select.value
    gt_opts = list(structure[f][m][t].keys())
    ground_truth_select.options = gt_opts
    ground_truth_select.value   = gt_opts[0]

@pn.depends(ground_truth_select.param.value, watch=True)
def _update_sample(event):
    f = feature_select.value
    m = model_select.value
    t = target_select.value
    gt = ground_truth_select.value
    ds = structure[f][m][t][gt]
    sample_select.options = ds
    sample_select.value   = ds[0]

selector_row = pn.Row(
    feature_select, model_select, target_select, ground_truth_select, sample_select,
    sizing_mode='stretch_width', margin=(10,10)
)

def load_and_precompute(folder):
    gt = pd.read_csv(f"{folder}/balanced_ground_truth.csv")
    inf = pd.read_csv(f"{folder}/balanced_inferred_network.csv")
    df = pd.concat([gt, inf], ignore_index=True)
    y_true = df["true_interaction"].values
    y_scores = df["Score"].values
    
    # Subsample every 20th score for the auroc and auprc
    idx = np.arange(0, len(y_scores), math.ceil(len(y_scores)*0.0001))
    y_true_ss   = y_true[idx]
    y_scores_ss = y_scores[idx]

    # ROC & PR
    fpr, tpr, _   = roc_curve(y_true_ss, y_scores_ss)
    prec, rec, _  = precision_recall_curve(y_true_ss, y_scores_ss)
    
    auroc = roc_auc_score(y_true_ss, y_scores_ss)
    auprc = average_precision_score(y_true_ss, y_scores_ss)
    
    # randomized uniform scores
    rand_scores        = np.random.uniform(0, 1, size=len(y_scores_ss))
    r_fpr, r_tpr, _    = roc_curve(y_true_ss, rand_scores)
    r_prec, r_rec, _   = precision_recall_curve(y_true_ss, rand_scores)
    rand_auroc         = roc_auc_score(y_true_ss, rand_scores)
    rand_auprc         = average_precision_score(y_true_ss, rand_scores)
    
    # histogram bins
    bins       = np.linspace(0, 1, 50)
    centers    = (bins[:-1] + bins[1:]) / 2
    idx        = np.clip(np.digitize(y_scores, bins) - 1, 0, len(centers)-1)
    tp_counts  = np.bincount(idx[y_true==1], minlength=len(centers))
    fp_counts  = np.bincount(idx[y_true==0], minlength=len(centers))
    
    thresh = 0.5
    raw = {
        "TP": y_scores[(y_true==1) & (y_scores>=thresh)],
        "FP": y_scores[(y_true==0) & (y_scores>=thresh)],
        "TN": y_scores[(y_true==0) & (y_scores< thresh)],
        "FN": y_scores[(y_true==1) & (y_scores< thresh)],
    }
    # compute boxplot stats
    box = {"class":[],"q1":[],"q2":[],"q3":[],"lower":[],"upper":[]}
    for cls, arr in raw.items():
        cls = str(cls)
        q1,q2,q3 = np.percentile(arr, [25,50,75])
        iqr      = q3 - q1
        low_wh   = max(arr.min(), q1-1.5*iqr)
        high_wh  = min(arr.max(), q3+1.5*iqr)
        box["class"].append(cls)
        box["q1"].append(q1)
        box["q2"].append(q2)
        box["q3"].append(q3)
        box["lower"].append(low_wh)
        box["upper"].append(high_wh)

    return {
        "bins":       bins,
        "centers":    centers,
        "tp_counts":  tp_counts,
        "fp_counts":  fp_counts,
        "total_pos":  int((y_true==1).sum()),
        "total_neg":  int((y_true==0).sum()),
        "roc":        dict(fpr=fpr, tpr=tpr),
        "pr":         dict(recall=rec, precision=prec),
        "rand_roc":   dict(fpr=r_fpr, tpr=r_tpr),
        "rand_pr":    dict(recall=r_rec, precision=r_prec),
        "box":        box,
        "auroc":      auroc,
        "auprc":      auprc,
        "rand_auroc": rand_auroc,
        "rand_auprc": rand_auprc,
    }

def create_loading_spinner():
    spinner = pn.indicators.LoadingSpinner(
        name='Loading', value=False, visible=False,
        width=50, height=50,
    )
    # apply inline styles so it always floats in the exact center:
    spinner.styles = {
        'position' : 'fixed',
        'top'      : '45%',
        'left'     : '40%',
        'transform': 'translate(-40%, -45%)',
        'z-index'  : '10000',
    }
    
    return spinner

class BoxAndWhiskerPlot:
    def __init__(self, data):
        self.source = ColumnDataSource(data["box"])
        self.box_fig = figure(
            x_range=["TP","FP","TN","FN"],
            width=400, height=300,
            title="Score Boxplot",
            tools=""
        )
        # draw the boxes and grab the renderer
        self.box_renderer = self.box_fig.vbar(
            x="class", width=0.7,
            top="q3", bottom="q1",
            source=self.source,
            fill_alpha=0.3,
            line_color="black",
        )

        # median line
        self.box_fig.segment(
            x0="class", y0="q2",
            x1="class", y1="q2",
            source=self.source,
            line_width=2,
            line_color="black"
        )

        # whiskers
        self.whisker = Whisker(source=self.source, base="class",
                        upper="upper", lower="lower")
        self.box_fig.add_layout(self.whisker)

        # 4) now create and add the hover tool for the boxes
        self.box_hover = HoverTool(
            renderers=[self.box_renderer],
            tooltips=[
                ("Class",       "@class"),
                ("Upper whisker","@upper{0.000}"),
                ("Q3",          "@q3{0.000}"),
                ("Median (Q2)", "@q2{0.000}"),
                ("Q1",          "@q1{0.000}"),
                ("Lower whisker", "@lower{0.000}")
            ]
        )
        self.box_fig.add_tools(self.box_hover)
    
class ROCCurve:
    def __init__(self, data):
        self.roc_source        = ColumnDataSource(data=data["roc"])
        self.rand_roc_source   = ColumnDataSource(data=data["rand_roc"])
        self.roc_threshold_line = Span(location=0.5, dimension='height', line_color='black', line_dash='dashed', line_width=2)
        self.roc_metric        = Title(text=f"AUROC: {data['auroc']:.3f}", align="center")
        self.rand_roc_metric   = Title(text=f"Random: {data['rand_auroc']:.3f}", align="center")
        self.roc_fig           = self.create_roc_curve()
    
    def create_roc_curve(self):
        roc_fig = figure(width=500, height=400, title="ROC Curve", x_axis_label="False Positive Rate", y_axis_label="True Positive Rate", tools="")
        roc_fig.line('fpr', 'tpr', source=self.roc_source, line_width=2, color='navy')
        roc_fig.line([0,1], [0,1], color='gray', line_dash='dashed')
        roc_fig.add_layout(self.roc_threshold_line)
        roc_fig.line(
            'fpr', 'tpr', source=self.rand_roc_source,
            line_dash='dashed', color='gray', line_width=2,
        )
        roc_fig.add_layout(self.roc_metric, 'below')
        roc_fig.add_layout(self.rand_roc_metric, 'below')
        
        roc_fig.x_range = Range1d(0, 1)
        roc_fig.y_range = Range1d(0, 1)
        
        return roc_fig
    
class PRCurve:
    def __init__(self, data):
        self.pr_source        = ColumnDataSource(data=data["pr"])
        
        self.rand_pr_source  = ColumnDataSource(data=data["rand_pr"])
        self.pr_threshold_line = Span(location=0.5, dimension='height', line_color='black', line_dash='dashed', line_width=2)
        self.pr_metric  = Title(text=f"AUPRC: {data['auprc']:.3f}", align="center")
        self.rand_pr_metric  = Title(text=f"Random: {data['rand_auprc']:.3f}", align="center")
        self.pr_fig           = self.create_pr_curve()
    
    def create_pr_curve(self):
        pr_fig = figure(width=500, height=400,
                title="Precision‑Recall Curve",
                x_axis_label="Recall", y_axis_label="Precision", tools="")
        pr_fig.line('recall', 'precision', source=self.pr_source, line_width=2, color='green')
        pr_fig.add_layout(self.pr_threshold_line)
        pr_fig.add_layout(self.pr_metric, 'below')
        pr_fig.add_layout(self.rand_pr_metric, 'below')
        pr_fig.line(
            'recall', 'precision', source=self.rand_pr_source,
            line_dash='dashed', color='gray', line_width=2,
        )
        
        pr_fig.x_range = Range1d(0, 1)
        pr_fig.y_range = Range1d(0, 1)
        
        return pr_fig

class Histogram:
    def __init__(self, data):
        self.hist_source = ColumnDataSource(data=dict(
                x   = data["centers"].tolist(),
                tp  = data["tp_counts"].tolist(),
                fp  = data["fp_counts"].tolist(),
                tn  = [0]*len(data["centers"]),
                fn  = [0]*len(data["centers"]),
            ))
        self.hist_threshold_line = Span(location=0.5, dimension='height', line_color='black', line_dash='dashed', line_width=2)
        self.hist_fig = self.create_histogram()
        
        
    def create_histogram(self):
        hist_fig = figure(width=600, height=300, title="Score Distribution", tools="")
        hist_fig.vbar(x='x', top='fp', width=0.015, color="#4195df", fill_alpha=0.7, source=self.hist_source, legend_label="Negative Score")
        hist_fig.vbar(x='x', top='tp', width=0.015, color="#dc8634", fill_alpha=0.7, source=self.hist_source, legend_label="Positive Score")

        hist_fig.add_layout(self.hist_threshold_line)
        
        legend = hist_fig.legend[0]
        hist_fig.add_layout(legend, 'below')
        legend.orientation       = 'horizontal'
        legend.label_text_align  = 'center'
        legend.margin = 0
        legend.padding = 5
        
        return hist_fig
        
# On startup, build the folder path from the three selectors and load metrics
initial_path = data_dir / feature_select.value / model_select.value / target_select.value / ground_truth_select.value / sample_select.value
print(f"Initial load from: {initial_path}")
data = load_and_precompute(initial_path)

# Create a loading spinner that will run when loading different datasets
spinner = create_loading_spinner()

# Create the TP, FP, TN, FN box and whisker plot
box_obj = BoxAndWhiskerPlot(data)
roc_curve_obj = ROCCurve(data)
pr_curve_obj = PRCurve(data)
hist_obj = Histogram(data)

# 2) Create your throttled slider in Python
slider = FloatSlider(
    start=0.0, end=1.0, value=0.5, step=0.005,
    name="Score Threshold"
)

# 3.5) Create a SECOND source to hold the un‐thresholded bin data
raw_source = ColumnDataSource(data=dict(
    centers = data["centers"].tolist(),
    raw_tp  = data["tp_counts"].tolist(),
    raw_fp  = data["fp_counts"].tolist(),
))

# this function will run *server‑side* whenever slider.value changes
@pn.depends(slider.param.value, watch=True)
def _update_spans(threshold):
    # threshold is your slider.value
    # compute the cut‐index on your raw arrays:
    centers   = raw_source.data['centers']
    tp_counts = raw_source.data['raw_tp']
    fp_counts = raw_source.data['raw_fp']
    k = next((i for i,c in enumerate(centers) if c>=threshold), len(centers))
    total_pos = sum(tp_counts)
    total_neg = sum(fp_counts)
    fp_above  = sum(fp_counts[k:])
    tp_above  = sum(tp_counts[k:])
    roc_curve_obj.roc_threshold_line.location   = fp_above/total_neg
    pr_curve_obj.pr_threshold_line.location     = tp_above/total_pos
    hist_obj.hist_threshold_line.location       = threshold

def update_dataset(event=None):
    # 1) show spinner immediately
    spinner.value   = True
    spinner.visible = True
    
    path = data_dir / feature_select.value / model_select.value / target_select.value / ground_truth_select.value / sample_select.value
    print(f'Loading data from: {path}')

    # 2) do the heavy lifting in a background thread
    def worker():
        try:
            d = load_and_precompute(path)
        except Exception as e:
            print("Error in load_and_precompute:", e)
            return
        
        # 3) schedule the UI updates under the doc lock
        def apply_update():
            print("Applying update")
            # histogram
            hist_obj.hist_source.data.update({
                "x":  d["centers"],
                "tp": d["tp_counts"],
                "fp": d["fp_counts"],
                "tn": [0]*len(d["centers"]),
                "fn": [0]*len(d["centers"]),
            })
            # ROC & PR
            roc_curve_obj.roc_source.data.update(d["roc"])
            roc_curve_obj.rand_roc_source.data.update(d["rand_roc"])
            
            pr_curve_obj.pr_source.data.update(d["pr"])
            pr_curve_obj.rand_pr_source.data.update(d["rand_pr"])
            
            raw_source.data.update({
                "centers": d["centers"],
                "raw_tp":  d["tp_counts"],
                "raw_fp":  d["fp_counts"],
            })
            
            # Boxplot
            box_obj.source.data.update(d["box"])
            
            # titles
            roc_curve_obj.roc_metric.text      = f"AUROC: {d['auroc']:.3f}"
            roc_curve_obj.rand_roc_metric.text = f"Random: {d['rand_auroc']:.3f}"
            pr_curve_obj.pr_metric.text      = f"AUPRC: {d['auprc']:.3f}"
            pr_curve_obj.rand_pr_metric.text = f"Random: {d['rand_auprc']:.3f}"
            
            # reset spans & slider
            slider.value                                = 0.5
            hist_obj.hist_threshold_line.location       = 0.5
            roc_curve_obj.roc_threshold_line.location   = 0.5
            pr_curve_obj.pr_threshold_line.location     = 0.5
            
            # hide spinner
            spinner.value   = False
            spinner.visible = False

        (doc or pn.state.curdoc).add_next_tick_callback(apply_update)

    try:
        t = threading.Thread(target=worker, daemon=True)
        t.start()
    except Exception as e:
        print("Failed to start worker thread:", e)


# Wire the Select to call update_dataset whenever it changes:
# dataset_select.param.watch(lambda ev: print("SELECTED →", ev.new), 'value')
model_select.param.watch(lambda ev: update_dataset(), 'value')
target_select.param.watch(lambda ev: update_dataset(), 'value')
ground_truth_select.param.watch(lambda ev: update_dataset(), 'value')
sample_select.param.watch(lambda ev: update_dataset(), 'value')

# Do the initial load:
update_dataset()

roc_pane  = pn.pane.Bokeh(roc_curve_obj.roc_fig,  sizing_mode='stretch_both')
pr_pane   = pn.pane.Bokeh(pr_curve_obj.pr_fig,   sizing_mode='stretch_both')
hist_pane = pn.pane.Bokeh(hist_obj.hist_fig, sizing_mode='stretch_both')
box_pane  = pn.pane.Bokeh(box_obj.box_fig,  sizing_mode='stretch_both')

# ─── 6) Panel layout ─────────────────────────────────────────────────────
dashboard = pn.Column(
    selector_row,                  # Panel Select
    pn.Row(roc_pane, pr_pane),       # Panel panes
    pn.Row(hist_pane, box_pane),
    slider,                          # Panel FloatSlider
)

pn.Column(dashboard, spinner).servable(
    title="AUROC + Histogram Dashboard"
)