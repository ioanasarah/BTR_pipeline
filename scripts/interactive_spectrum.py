"""Interactive mean mass spectrum using plotly -- zoom in to see point spacing."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import spatialdata as sd

ZARR_PATH = Path(
    "c:/Users/P70078823/Desktop/Ioana BTR/data/spatialdata_test/1 1hnr.zarr"
)


def main():
    sdata = sd.read_zarr(str(ZARR_PATH))
    table_name = list(sdata.tables.keys())[0]
    table = sdata.tables[table_name]

    # Get m/z values
    for col in ["mz", "m/z", "mass"]:
        if col in table.var.columns:
            mz = table.var[col].values.astype(float)
            break
    else:
        # Fallback: reconstruct from info metadata if index is mz_0, mz_1...
        n = table.shape[1]
        mz = np.linspace(100, 2000, n)
        print(f"Reconstructed m/z axis: {n} points from 100-2000 Da")

    mean_intensities = np.asarray(table.X.mean(axis=0)).ravel()

    # Compute spacing stats
    spacing = np.diff(mz)
    print(f"m/z range: {mz[0]:.2f} - {mz[-1]:.2f} Da")
    print(f"Number of points: {len(mz)}")
    print(f"Spacing: min={spacing.min():.6f}, max={spacing.max():.6f}, "
          f"mean={spacing.mean():.6f} Da")

    fig = go.Figure()

    # Line trace
    fig.add_trace(go.Scattergl(
        x=mz,
        y=mean_intensities,
        mode="lines",
        line=dict(width=0.5, color="black"),
        name="Mean spectrum",
    ))

    # Markers (visible when zoomed in)
    fig.add_trace(go.Scattergl(
        x=mz,
        y=mean_intensities,
        mode="markers",
        marker=dict(size=3, color="red", opacity=0.6),
        name="Data points",
        visible="legendonly",
    ))

    fig.update_layout(
        title=f"Mean mass spectrum -- {ZARR_PATH.stem} "
              f"({len(mz)} points, ~{spacing.mean():.4f} Da spacing)",
        xaxis_title="m/z (Da)",
        yaxis_title="Mean intensity",
        hovermode="x unified",
        template="plotly_white",
    )

    out = ZARR_PATH.parent / "diagnostics" / "interactive_spectrum.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out))
    print(f"\nSaved: {out}")
    print("Open in browser to zoom and inspect point spacing.")
    print("Click 'Data points' in legend to toggle markers.")


if __name__ == "__main__":
    main()
