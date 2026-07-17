$ErrorActionPreference = "Stop"
$base = Split-Path -Parent $MyInvocation.MyCommand.Path

$papers = @(
    @{ Category = "01_planning_search_vs_policy"; Id = "1705.08439"; Name = "Expert-Iteration" },
    @{ Category = "01_planning_search_vs_policy"; Id = "1712.01815"; Name = "AlphaZero" },
    @{ Category = "01_planning_search_vs_policy"; Id = "1911.08265"; Name = "MuZero" },
    @{ Category = "01_planning_search_vs_policy"; Id = "1707.06203"; Name = "Imagination-Augmented-Agents" },
    @{ Category = "01_planning_search_vs_policy"; Id = "1511.06295"; Name = "Policy-Distillation" },
    @{ Category = "01_planning_search_vs_policy"; Id = "2005.07404"; Name = "Planning-Learning-Compute-Tradeoff" },
    @{ Category = "01_planning_search_vs_policy"; Id = "1906.08649"; Name = "POPLIN" },
    @{ Category = "01_planning_search_vs_policy"; Id = "2204.03597"; Name = "IMPLANT" },
    @{ Category = "01_planning_search_vs_policy"; Id = "1805.10755"; Name = "Dual-Policy-Iteration" },
    @{ Category = "01_planning_search_vs_policy"; Id = "2606.16286"; Name = "FlowMPC" },
    @{ Category = "02_model_based_vs_model_free"; Id = "1906.08253"; Name = "MBPO" },
    @{ Category = "02_model_based_vs_model_free"; Id = "1803.00101"; Name = "Model-Based-Value-Expansion" },
    @{ Category = "02_model_based_vs_model_free"; Id = "1805.12114"; Name = "PETS" },
    @{ Category = "02_model_based_vs_model_free"; Id = "1907.02057"; Name = "Benchmarking-MBRL" },
    @{ Category = "03_test_time_compute_vs_model_capacity"; Id = "2408.03314"; Name = "Scaling-Test-Time-Compute" },
    @{ Category = "03_test_time_compute_vs_model_capacity"; Id = "2407.21787"; Name = "Large-Language-Monkeys" },
    @{ Category = "03_test_time_compute_vs_model_capacity"; Id = "2501.19393"; Name = "s1-Test-Time-Scaling" },
    @{ Category = "03_test_time_compute_vs_model_capacity"; Id = "2407.06023"; Name = "Distilling-System-2-into-System-1" },
    @{ Category = "03_test_time_compute_vs_model_capacity"; Id = "2203.11171"; Name = "Self-Consistency" },
    @{ Category = "04_fixed_budget_scaling"; Id = "2203.15556"; Name = "Chinchilla" },
    @{ Category = "04_fixed_budget_scaling"; Id = "1905.11946"; Name = "EfficientNet" },
    @{ Category = "04_fixed_budget_scaling"; Id = "2001.08361"; Name = "Scaling-Laws" },
    @{ Category = "04_fixed_budget_scaling"; Id = "2107.05407"; Name = "PonderNet" },
    @{ Category = "04_fixed_budget_scaling"; Id = "1603.08983"; Name = "Adaptive-Computation-Time" }
)

foreach ($paper in $papers) {
    $dir = Join-Path $base $paper.Category
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $output = Join-Path $dir ($paper.Id + "_" + $paper.Name + ".pdf")
    if (Test-Path -LiteralPath $output) {
        $file = Get-Item -LiteralPath $output
        $header = if ($file.Length -ge 5) {
            [System.Text.Encoding]::ASCII.GetString([System.IO.File]::ReadAllBytes($output), 0, 5)
        } else {
            ""
        }
        if ($file.Length -gt 100KB -and $header -eq "%PDF-") {
            Write-Host "skip $output"
            continue
        }
        Write-Host "replace invalid or incomplete file $output"
    }

    $uri = "https://arxiv.org/pdf/" + $paper.Id
    Write-Host "download $uri -> $output"
    Invoke-WebRequest -UseBasicParsing -Uri $uri -OutFile $output -TimeoutSec 120
}
