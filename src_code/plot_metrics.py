import matplotlib.pyplot as plt
import numpy as np

def plot_metrics_grid(df, model_col="model", horizon_col="horizon_h",
                      models=None, horizons=None,
                      metrics=(("MAE", "MAE"), ("MSE", "MSE"), ("R2", "R\u00b2")),
                      val_prefix="val_", test_prefix="test_",
                      unit="", savepath=None):
    """
    Vẽ lưới chỉ số kiểu Fig.4 (SA-TCN-DeepAR).
      Hàng  = từng metric trong `metrics` (mỗi phần tử: (hậu-tố-cột, nhãn hiển thị))
      Cột   = Validation | Test
      Trục x = các horizon; mỗi đường = một mô hình.
    df cần có cột: {model_col}, {horizon_col}, và {val_/test_}{metric} cho mỗi metric.
    """
    if models is None:   models   = list(df[model_col].unique())
    if horizons is None: horizons = sorted(df[horizon_col].unique())

    splits  = [(val_prefix, "Validation"), (test_prefix, "Test")]
    markers = ["o", "^", "s", "D", "v", "*"]
    labels  = "abcdefghijkl"
    nrows   = len(metrics)

    fig, axes = plt.subplots(nrows, 2, figsize=(9, 3.0 * nrows), sharex=True)
    axes = np.atleast_2d(axes)
    k = 0
    for r, (suf, disp) in enumerate(metrics):
        for c, (pref, split) in enumerate(splits):
            ax = axes[r, c]
            col = f"{pref}{suf}"
            for mi, mdl in enumerate(models):
                sub = df[df[model_col] == mdl].sort_values(horizon_col)
                ax.plot(sub[horizon_col].values, sub[col].values,
                        marker=markers[mi % len(markers)], markersize=6,
                        linewidth=1.6, label=mdl)
            ax.set_xticks(horizons)
            ax.set_xticklabels([f"{h}h" for h in horizons])
            ylab = f"{split} {disp}" + (f" /{unit}" if suf != "R2" else "")
            ax.set_ylabel(ylab, fontsize=10)
            ax.set_xlabel("Prediction duration", fontsize=10)
            ax.grid(True, alpha=0.3, linewidth=0.7)
            ax.set_title(f"({labels[k]}) {disp} value of {split}", fontsize=10)
            ax.legend(fontsize=8, ncol=len(models), loc="best", frameon=False)
            k += 1

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    return fig