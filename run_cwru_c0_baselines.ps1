# CWRU C0 目标工况基础消融实验（5-way 1-shot）
#
# 实验组合：
#   1. STFT-CNN4-MAML
#   2. STFT-LSKLite-MAML
#   3. STFT-LSKLite-GCNet-MAML
# 每个模型依次运行 seed 3、24、38，共 9 次完整 train + test。
#
# 脚本会临时修改 config_cwru.py，结束时自动恢复原配置，并在需要时
# 重新生成原目标工况对应的预处理数据，避免配置与 processed 数据错位。

$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$ConfigPath = Join-Path $ProjectRoot 'config_cwru.py'
$BackupPath = Join-Path ([IO.Path]::GetTempPath()) (
    'config_cwru_' + [Guid]::NewGuid().ToString('N') + '.py'
)
$LogRoot = Join-Path $ProjectRoot 'logs'
$ModelRoot = Join-Path $ProjectRoot 'model_save'
$ResultRoot = Join-Path $ProjectRoot 'results\cwru'
$SummaryPath = Join-Path $ResultRoot 'CWRU-C0-baselines-5way1shot-summary.csv'

New-Item -ItemType Directory -Force -Path $LogRoot, $ModelRoot, $ResultRoot | Out-Null

if (-not (Test-Path $ConfigPath -PathType Leaf)) {
    throw "找不到配置文件：$ConfigPath"
}
if (-not (Test-Path (Join-Path $ProjectRoot 'CWRU\__file__\metadata.txt') -PathType Leaf)) {
    throw "找不到CWRU原始数据：$ProjectRoot\CWRU\__file__\metadata.txt"
}

Copy-Item -LiteralPath $ConfigPath -Destination $BackupPath -Force
$OriginalConfig = [IO.File]::ReadAllText($BackupPath, [Text.Encoding]::UTF8)
$OriginalTargetMatch = [regex]::Match(
    $OriginalConfig,
    "'target_condition'\s*:\s*'([^']+)'"
)
if (-not $OriginalTargetMatch.Success) {
    throw '无法从config_cwru.py读取原目标工况。'
}
$OriginalTarget = $OriginalTargetMatch.Groups[1].Value

function Set-CwruExperimentConfig {
    param(
        [Parameter(Mandatory = $true)][string]$Backbone,
        [Parameter(Mandatory = $true)][string]$Attention,
        [Parameter(Mandatory = $true)][int]$Seed
    )

    # 必须显式按UTF-8读取；Windows PowerShell 5.1会把无BOM UTF-8
    # 按系统代码页解释，中文注释可能吞掉相邻ASCII引号并破坏Python语法。
    $Text = [IO.File]::ReadAllText($ConfigPath, [Text.Encoding]::UTF8)
    $Edits = @(
        @{
            Pattern = "'source_condition'\s*:\s*\[[^\]]*\]\s*,"
            Replacement = "'source_condition': ['C1_1772rpm', 'C2_1750rpm', 'C3_1730rpm'],"
        },
        @{
            Pattern = "'target_condition'\s*:\s*'[^']+'\s*,"
            Replacement = "'target_condition': 'C0_1797rpm',"
        },
        @{
            Pattern = "'n_way'\s*:\s*\d+\s*,"
            Replacement = "'n_way': 5,"
        },
        @{
            Pattern = "'k_shot'\s*:\s*\d+\s*,"
            Replacement = "'k_shot': 1,"
        },
        @{
            Pattern = "'q_query'\s*:\s*\d+\s*,"
            Replacement = "'q_query': 15,"
        },
        @{
            Pattern = "'backbone'\s*:\s*'[^']+'\s*,"
            Replacement = "'backbone': '$Backbone',"
        },
        @{
            Pattern = "'frequency_module'\s*:\s*'[^']+'\s*,"
            Replacement = "'frequency_module': 'none',"
        },
        @{
            Pattern = "'denoise_module'\s*:\s*'[^']+'\s*,"
            Replacement = "'denoise_module': 'none',"
        },
        @{
            Pattern = "'attention_module'\s*:\s*'[^']+'\s*,"
            Replacement = "'attention_module': '$Attention',"
        },
        @{
            Pattern = "'task_weighting_mode'\s*:\s*'[^']+'\s*,"
            Replacement = "'task_weighting_mode': 'none',"
        },
        @{
            Pattern = "'seed'\s*:\s*\d+\s*,"
            Replacement = "'seed': $Seed,"
        }
    )

    foreach ($Edit in $Edits) {
        $MatchCount = [regex]::Matches($Text, $Edit.Pattern).Count
        if ($MatchCount -ne 1) {
            throw "配置字段匹配数量异常：$($Edit.Pattern)，实际匹配$MatchCount处。"
        }
        $Text = [regex]::Replace($Text, $Edit.Pattern, $Edit.Replacement)
    }

    $Utf8NoBom = [Text.UTF8Encoding]::new($false)
    [IO.File]::WriteAllText($ConfigPath, $Text, $Utf8NoBom)
}

# Windows PowerShell 5.1 会把原生程序的 stderr 包装成 ErrorRecord。
# 在全局 Stop 模式下直接使用 2>&1 | Tee-Object，会在 traceback 第一行就终止
# 管道。这里只在 Python 进程运行期间临时改为 Continue，并仍以退出码判定成败。
$script:LastPythonExitCode = -1
function Invoke-PythonLogged {
    param(
        [Parameter(Mandatory = $true)][string[]]$PythonArguments,
        [Parameter(Mandatory = $true)][string]$LogPath
    )

    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & python -u @PythonArguments 2>&1 | Tee-Object -FilePath $LogPath
        $script:LastPythonExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
}

$Experiments = @(
    [PSCustomObject]@{
        Model = 'STFT-CNN4-MAML'
        Backbone = 'cnn4'
        Attention = 'none'
    },
    [PSCustomObject]@{
        Model = 'STFT-LSKLite-MAML'
        Backbone = 'lsk_lite'
        Attention = 'none'
    },
    [PSCustomObject]@{
        Model = 'STFT-LSKLite-GCNet-MAML'
        Backbone = 'lsk_lite'
        Attention = 'gcnet'
    }
)
$Seeds = @(3, 24, 38)
$Summary = [System.Collections.Generic.List[object]]::new()
$C0Prepared = $false
$FatalError = $null

try {
    # 三种模型共用相同的C0数据划分，只需预处理一次。
    Set-CwruExperimentConfig -Backbone 'cnn4' -Attention 'none' -Seed 3

    Write-Host ''
    Write-Host '============================================================'
    Write-Host '正在生成C0目标工况预处理数据'
    Write-Host "Source: C1_1772rpm, C2_1750rpm, C3_1730rpm"
    Write-Host "Target: C0_1797rpm"
    Write-Host '============================================================'

    $PreprocessLog = Join-Path $LogRoot 'CWRU-C0-preprocess.log'
    Invoke-PythonLogged `
        -PythonArguments @('.\run_cwru_maml.py', '--preprocess', '--clean') `
        -LogPath $PreprocessLog
    if ($script:LastPythonExitCode -ne 0) {
        throw "C0预处理失败，详见：$PreprocessLog"
    }
    $C0Prepared = $true

    foreach ($Experiment in $Experiments) {
        foreach ($Seed in $Seeds) {
            Set-CwruExperimentConfig `
                -Backbone $Experiment.Backbone `
                -Attention $Experiment.Attention `
                -Seed $Seed

            $RunId = "CWRU-$($Experiment.Model)-C0-seed$Seed"
            $ModelPath = Join-Path $ModelRoot $RunId
            $LogPath = Join-Path $LogRoot ($RunId + '.log')

            Write-Host ''
            Write-Host '============================================================'
            Write-Host "开始实验：$RunId"
            Write-Host '任务：5-way 1-shot 15-query'
            Write-Host '任务加权：none'
            Write-Host '============================================================'

            Invoke-PythonLogged `
                -PythonArguments @(
                    '.\run_cwru_maml.py',
                    '--train',
                    '--test',
                    '--model_path',
                    $ModelPath
                ) `
                -LogPath $LogPath
            $RunExitCode = $script:LastPythonExitCode

            $Accuracy = $null
            $AccuracyText = 'MISSING'
            $AccuracyMatch = Select-String `
                -LiteralPath $LogPath `
                -Pattern 'Meta Test Accuracy:\s*([0-9]*\.?[0-9]+)' |
                Select-Object -Last 1

            if ($null -ne $AccuracyMatch) {
                $Accuracy = [double]::Parse(
                    $AccuracyMatch.Matches[0].Groups[1].Value,
                    [Globalization.CultureInfo]::InvariantCulture
                ) * 100.0
                $AccuracyText = '{0:F2}%' -f $Accuracy
            }

            $Status = if ($RunExitCode -eq 0 -and $null -ne $Accuracy) {
                'OK'
            } elseif ($RunExitCode -ne 0) {
                "FAILED($RunExitCode)"
            } else {
                'MISSING'
            }

            $Summary.Add([PSCustomObject]@{
                Dataset = 'CWRU'
                Model = $Experiment.Model
                Seed = $Seed
                Target = 'C0_1797rpm'
                Way = 5
                Shot = 1
                TestAccuracy = $AccuracyText
                TestAccuracyPercent = $Accuracy
                Status = $Status
                RunId = $RunId
                Log = $LogPath
            })
        }
    }
}
catch {
    $FatalError = $_
}
finally {
    Copy-Item -LiteralPath $BackupPath -Destination $ConfigPath -Force
    Remove-Item -LiteralPath $BackupPath -Force

    # 若脚本开始前不是C0目标，则恢复原目标对应的processed布局。
    if ($C0Prepared -and $OriginalTarget -ne 'C0_1797rpm') {
        Write-Host ''
        Write-Host "正在恢复原目标工况 $OriginalTarget 的预处理数据……"
        $RestoreLog = Join-Path $LogRoot ("CWRU-restore-$OriginalTarget.log")
        Invoke-PythonLogged `
            -PythonArguments @('.\run_cwru_maml.py', '--preprocess', '--clean') `
            -LogPath $RestoreLog
        if ($script:LastPythonExitCode -ne 0) {
            Write-Warning "原目标工况预处理数据恢复失败，请查看：$RestoreLog"
        }
    }
}

Write-Host ''
Write-Host '========================================================================'
Write-Host '                    CWRU C0 全部实验测试准确率'
Write-Host '========================================================================'

if ($Summary.Count -gt 0) {
    $DisplaySummary = $Summary | Select-Object `
        Model, Seed, Target, Way, Shot, TestAccuracy, Status
    $DisplaySummary | Format-Table -AutoSize
    $Summary | Export-Csv -LiteralPath $SummaryPath -NoTypeInformation -Encoding utf8

    $ValidResults = @($Summary | Where-Object { $null -ne $_.TestAccuracyPercent })
    if ($ValidResults.Count -gt 0) {
        $OverallMean = ($ValidResults |
            Measure-Object -Property TestAccuracyPercent -Average).Average
        Write-Host ('有效实验：{0}/9；总体平均准确率：{1:F2}%' -f `
            $ValidResults.Count, $OverallMean)

        foreach ($Experiment in $Experiments) {
            $ModelResults = @($ValidResults | Where-Object {
                $_.Model -eq $Experiment.Model
            })
            if ($ModelResults.Count -gt 0) {
                $ModelMean = ($ModelResults |
                    Measure-Object -Property TestAccuracyPercent -Average).Average
                Write-Host ('{0}：{1:F2}%（{2}/3 seeds）' -f `
                    $Experiment.Model, $ModelMean, $ModelResults.Count)
            }
        }
    }
    Write-Host "汇总CSV：$SummaryPath"
} else {
    Write-Host '没有产生可汇总的实验结果。'
}

Write-Host '========================================================================'

if ($null -ne $FatalError) {
    Write-Error $FatalError
    exit 1
}
