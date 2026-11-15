# Matched OSM x accident-history ablation for all six models (Delaware, Node2Vec on).
#
#   OSM      : off | all (the five groups of the OSM experiment)
#   History  : none | all
#   Models   : mlp, gcn, gat, graphsage, gatedgcn, graph_transformer
#   Tasks    : classification, regression
#   = 2 x 2 x 6 x 2 = 48 runs, 3 seeds each, 100 epochs (train.py defaults).
#
# All four cells use the same months: when the history archive is present, even
# --history-mode none keeps only months with usable history. Do not compare these
# runs with the older results/DE_* folders (those used every month).
#
# Run from the repository root:   powershell -ExecutionPolicy Bypass -File scripts\run_feature_ablation.ps1
# Only one factor:                 ... -Osm off          (history comparison only, 24 runs)
#                                  ... -History none     (OSM comparison only, 24 runs)
# Safe to re-run: --resume skips finished seeds and continues interrupted ones.

param(
    [ValidateSet("both", "off", "all")] [string]$Osm = "both",
    [ValidateSet("both", "none", "all")] [string]$History = "both",
    [string[]]$Models = @("mlp", "gcn", "gat", "graphsage", "gatedgcn", "graph_transformer"),
    [string[]]$Tasks = @("classification", "regression"),
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$osmLevels = if ($Osm -eq "both") { @("off", "all") } else { @($Osm) }
$historyLevels = if ($History -eq "both") { @("none", "all") } else { @($History) }

# Preflight: fail now rather than after hours of training.
if (-not (Test-Path "data\accident_history\outputs\manifest.json")) {
    throw "History archive missing: data\accident_history\outputs\manifest.json (get it from the history teammate)."
}
if ($osmLevels -contains "all") {
    if (-not (Test-Path "data\DE_osm_features.parquet")) {
        throw "OSM table missing: data\DE_osm_features.parquet (build with osm_features_and_eda.ipynb or copy it)."
    }
    & $Python -c "import pyarrow" 2>$null
    if ($LASTEXITCODE -ne 0) { throw "pyarrow is not installed: run  $Python -m pip install pyarrow" }
}

$failed = @()
foreach ($m in $Models) {
    foreach ($t in $Tasks) {
        foreach ($o in $osmLevels) {
            foreach ($h in $historyLevels) {
                $name = "feat_DE_${m}_${t}_n2v_osm_${o}_hist_${h}"
                $cmd = @("-m", "src.model.train", "--state", "DE", "--model", $m, "--task", $t,
                         "--history-mode", $h, "--run-name", $name, "--resume")
                if ($o -eq "all") { $cmd += @("--osm-groups", "all") }
                Write-Host "`n=== $name ===" -ForegroundColor Cyan
                & $Python @cmd
                if ($LASTEXITCODE -ne 0) { $failed += $name; Write-Warning "$name failed (exit $LASTEXITCODE)" }
            }
        }
    }
}

if ($failed) { Write-Warning "Failed runs:`n  $($failed -join "`n  ")"; exit 1 }
Write-Host "`nAll runs finished. Results: results\feat_DE_*" -ForegroundColor Green
