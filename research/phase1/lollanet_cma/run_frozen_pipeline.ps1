param(
    [string]$Python = 'D:\Coding\LollaNet\research\phase1\lollasteer_q\.venv\Scripts\python.exe',
    [string]$DataRoot = 'D:\LollaNetData\lollanet-cma'
)

$ErrorActionPreference = 'Stop'
$qualificationView = Join-Path $DataRoot 'access-views\qualification-v1'
$vectorTrainView = Join-Path $DataRoot 'access-views\vector-train-v1'
$developmentView = Join-Path $DataRoot 'access-views\development-v1'
$readiness = Join-Path $DataRoot 'qualification\qualification-readiness-v7.json'
$runs = Join-Path $DataRoot 'qualification\runs-v3'

function Invoke-Python {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Read-RequiredJson {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required artifact is missing: $Path"
    }
    return Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
}

Write-Output 'PIPELINE_V1 stage=qualification'
$candidateIds = @('qwen3-4b-instruct-2507', 'qwen2.5-3b-instruct', 'phi-3.5-mini-instruct')
foreach ($candidateId in $candidateIds) {
    $runDir = Join-Path $runs $candidateId
    $manifestPath = Join-Path $runDir 'run_manifest.json'
    if (-not (Test-Path -LiteralPath $manifestPath)) {
        Write-Output "PIPELINE_V1 running_candidate=$candidateId"
        Invoke-Python -Arguments @(
            '-m', 'research.phase1.lollanet_cma.qualification.run_candidate',
            '--access-view', $qualificationView,
            '--readiness', $readiness,
            '--candidate', $candidateId,
            '--output', $runDir
        )
    }
    $manifest = Read-RequiredJson -Path $manifestPath
    if (-not $manifest.hard_gate_pass -or $manifest.record_count -ne 200) {
        throw "Qualification candidate did not complete its hard gates: $candidateId"
    }
}

$selection = Join-Path $DataRoot 'qualification\base_model_qualification-v3.json'
Write-Output 'PIPELINE_V1 stage=base_model_selection'
Invoke-Python -Arguments @(
    '-m', 'research.phase1.lollanet_cma.qualification.select_candidate',
    '--run-dir', (Join-Path $runs $candidateIds[0]),
    '--run-dir', (Join-Path $runs $candidateIds[1]),
    '--run-dir', (Join-Path $runs $candidateIds[2]),
    '--output', $selection
)

$directionPreflight = Join-Path $DataRoot 'directions\direction-preflight-v3.json'
$directionOutput = Join-Path $DataRoot 'directions\fit-v2'
Write-Output 'PIPELINE_V1 stage=direction_preflight'
Invoke-Python -Arguments @(
    '-m', 'research.phase1.lollanet_cma.directions.audit_preflight',
    '--access-view', $vectorTrainView,
    '--selection', $selection,
    '--output', $directionPreflight
)
$directionAudit = Read-RequiredJson -Path $directionPreflight
if ($directionAudit.direction_gate_status -ne 'ready') {
    throw "Direction preflight is not ready: $($directionAudit.direction_gate_status)"
}

Write-Output 'PIPELINE_V1 stage=direction_fitting'
& $Python -m research.phase1.lollanet_cma.directions.fit_directions `
    --access-view $vectorTrainView `
    --selection $selection `
    --preflight $directionPreflight `
    --output $directionOutput
if ($LASTEXITCODE -eq 2) {
    throw 'Direction fitting completed but a preregistered sufficiency gate failed'
}
if ($LASTEXITCODE -ne 0) {
    throw "Direction fitting failed with exit code $LASTEXITCODE"
}
$directionManifest = Join-Path $directionOutput 'run_manifest.json'
$directionRun = Read-RequiredJson -Path $directionManifest
if (-not $directionRun.hard_gate_pass -or $directionRun.status -ne 'fitted') {
    throw "Direction run did not pass hard gates: $($directionRun.status)"
}

$developmentPreflight = Join-Path $DataRoot 'development\development-preflight-v2.json'
$developmentOutput = Join-Path $DataRoot 'development\run-v1'
Write-Output 'PIPELINE_V1 stage=development_preflight'
Invoke-Python -Arguments @(
    '-m', 'research.phase1.lollanet_cma.development.audit_preflight',
    '--access-view', $developmentView,
    '--direction-manifest', $directionManifest,
    '--output', $developmentPreflight
)
$developmentAudit = Read-RequiredJson -Path $developmentPreflight
if ($developmentAudit.development_gate_status -ne 'ready') {
    throw "Development preflight is not ready: $($developmentAudit.development_gate_status)"
}

Write-Output 'PIPELINE_V1 stage=development_selection'
& $Python -m research.phase1.lollanet_cma.development.run_development `
    --access-view $developmentView `
    --direction-manifest $directionManifest `
    --preflight $developmentPreflight `
    --output $developmentOutput
if ($LASTEXITCODE -eq 2) {
    throw 'Development selection completed but the preregistered specificity gate failed'
}
if ($LASTEXITCODE -ne 0) {
    throw "Development selection failed with exit code $LASTEXITCODE"
}
$developmentManifest = Join-Path $developmentOutput 'run_manifest.json'
$developmentRun = Read-RequiredJson -Path $developmentManifest
if (-not $developmentRun.hard_gate_pass -or $developmentRun.status -ne 'frozen') {
    throw "Development did not freeze a passing selection: $($developmentRun.status)"
}

$stage0 = Join-Path $DataRoot 'stage0\bundle-v4'
$releasePreflight = Join-Path $DataRoot 'evaluation\sealed-release-preflight-v2.json'
$evaluationRelease = Join-Path $DataRoot 'evaluation\evaluation-release-v1.json'
Write-Output 'PIPELINE_V1 stage=sealed_release_preflight'
Invoke-Python -Arguments @(
    '-m', 'research.phase1.lollanet_cma.evaluation.audit_release',
    '--stage0-bundle', $stage0,
    '--development-manifest', $developmentManifest,
    '--output', $releasePreflight
)
$releaseAudit = Read-RequiredJson -Path $releasePreflight
if ($releaseAudit.sealed_release_gate_status -ne 'ready' -or -not $releaseAudit.unseal_authorized) {
    throw "Sealed release preflight is not ready: $($releaseAudit.sealed_release_gate_status)"
}

Write-Output 'PIPELINE_V1 stage=sealed_release'
Invoke-Python -Arguments @(
    '-m', 'research.phase1.lollanet_cma.evaluation.release',
    '--stage0-bundle', $stage0,
    '--preflight', $releasePreflight,
    '--output-manifest', $evaluationRelease
)

$probeOutput = Join-Path $DataRoot 'evaluation\probe-v1'
$probeManifest = Join-Path $probeOutput 'run_manifest.json'
Write-Output 'PIPELINE_V1 stage=probe_preparation'
Invoke-Python -Arguments @(
    '-m', 'research.phase1.lollanet_cma.sealed_eval.prepare_probe',
    '--development-manifest', $developmentManifest,
    '--vector-train-view', $vectorTrainView,
    '--development-view', $developmentView,
    '--output', $probeOutput
)

$sealedPreflight = Join-Path $DataRoot 'evaluation\sealed-evaluation-preflight-v2.json'
Write-Output 'PIPELINE_V1 stage=sealed_evaluation_preflight'
Invoke-Python -Arguments @(
    '-m', 'research.phase1.lollanet_cma.sealed_eval.audit_preflight',
    '--evaluation-release', $evaluationRelease,
    '--probe-manifest', $probeManifest,
    '--output', $sealedPreflight
)
$sealedAudit = Read-RequiredJson -Path $sealedPreflight
if ($sealedAudit.sealed_evaluation_gate_status -ne 'ready' -or -not $sealedAudit.evaluation_authorized) {
    throw "Sealed evaluation preflight is not ready: $($sealedAudit.sealed_evaluation_gate_status)"
}

$rawOutput = Join-Path $DataRoot 'evaluation\raw-run-v1'
Write-Output 'PIPELINE_V1 stage=sealed_raw_evaluation'
Invoke-Python -Arguments @(
    '-m', 'research.phase1.lollanet_cma.sealed_eval.run_evaluation',
    '--preflight', $sealedPreflight,
    '--output', $rawOutput
)

$reportOutput = Join-Path $DataRoot 'evaluation\report-v1'
Write-Output 'PIPELINE_V1 stage=report_assembly'
Invoke-Python -Arguments @(
    '-m', 'research.phase1.lollanet_cma.sealed_eval.assemble_report',
    '--raw-manifest', (Join-Path $rawOutput 'run_manifest.json'),
    '--output', $reportOutput
)
Write-Output 'PIPELINE_V1 complete=report_assembled'
