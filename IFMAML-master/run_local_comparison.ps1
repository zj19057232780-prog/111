param(
    [ValidateSet('cwru', 'pu')][string]$Dataset = 'cwru',
    [ValidateSet('protonet', 'matchingnet', 'relationnet', 'maml', 'resnet18', 'resnet18_ft', 'tl_wdcnn')][string]$Method = 'protonet',
    [ValidateSet(1, 3, 5)][int]$Shot = 1,
    [ValidateSet('auto', 'cuda', 'cpu')][string]$Device = 'auto',
    [ValidateRange(1, 100000)][int]$Epochs = 500,
    [string]$Python = 'C:\ANACONDA\envs\ifmaml2\python.exe',
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python not found: $Python. Set -Python to the project environment executable."
}
$runnerArgs = @('-u', (Join-Path $PSScriptRoot 'run_table_row.py'),
    '--project', $PSScriptRoot, '--dataset', $Dataset, '--method', $Method,
    '--shot', "$Shot", '--device', $Device, '--epochs', "$Epochs")
if ($CheckOnly) { $runnerArgs += '--check-only' }
& $Python @runnerArgs
exit $LASTEXITCODE
