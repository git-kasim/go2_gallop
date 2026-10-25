# Isaac Sim 6.1 を、外からスクリプトを送れる状態で起動する。
# 拡張 isaacsim.code_editor.python_server が 127.0.0.1:8226 で待ち受ける（ループバックのみ、認証なし）。
# ウィンドウを最小化すると動画のコマが撮れずに止まるので、最小化しないこと。
#
# 使い方（パスは例。自分の環境の場所に置き換える）:
#   powershell -ExecutionPolicy Bypass -File launch_isaacsim.ps1 -Venv C:\env_isaacsim                  # pip 版の venv を指定
#   powershell -ExecutionPolicy Bypass -File launch_isaacsim.ps1 -Exe C:\isaac-sim\isaac-sim.bat        # 実行ファイルを直接指定（standalone 版など）
# -Venv も -Exe も省くと環境変数 ISAACSIM_VENV を使う。

param(
    [string]$Venv = $env:ISAACSIM_VENV,
    [string]$Exe = ""
)

if (-not $Exe) {
    if (-not $Venv) {
        Write-Error "Specify -Venv <pip venv>, -Exe <isaacsim.exe / isaac-sim.bat>, or set ISAACSIM_VENV."
        exit 1
    }
    $Exe = Join-Path $Venv "Scripts\isaacsim.exe"
}
if (-not (Test-Path $Exe)) {
    Write-Error "Isaac Sim not found: $Exe  (use -Venv <pip venv> or -Exe <isaacsim.exe / isaac-sim.bat>)"
    exit 1
}

Start-Process -FilePath $Exe -ArgumentList @("--enable", "isaacsim.code_editor.python_server")
Write-Host "Isaac Sim launched: $Exe (python_server on 127.0.0.1:8226)."
