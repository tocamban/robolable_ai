$port = 5050
$killed = $false
try {
    $conns = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($conns) {
        $pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($p in $pids) {
            try {
                $proc = Get-Process -Id $p -ErrorAction Stop
                Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
                Write-Host ("[+] Đã dừng tiến trình: " + $proc.ProcessName + " (PID: " + $p + ")") -ForegroundColor Green
                $killed = $true
            } catch {}
        }
    }
} catch {}

# Đồng thời quét các tiến trình python chạy app.py nếu có
try {
    $pyProcs = Get-CimInstance Win32_Process -Filter "Name LIKE 'python%'" -ErrorAction SilentlyContinue
    foreach ($p in $pyProcs) {
        if ($p.CommandLine -like "*app.py*" -and $p.CommandLine -like "*robolabel_app*") {
            try {
                Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
                Write-Host ("[+] Đã dừng tiến trình python app.py (PID: " + $p.ProcessId + ")") -ForegroundColor Green
                $killed = $true
            } catch {}
        }
    }
} catch {}

if ($killed) {
    Write-Host "`n[✓] SERVER ROBOLABEL AI ĐÃ ĐƯỢC TẮT HOÀN TOÀN!" -ForegroundColor Green
    Write-Host "[*] Cổng 5050 đã được giải phóng sạch sẽ.`n" -ForegroundColor Cyan
} else {
    Write-Host "`n[*] Không tìm thấy server RoboLabel nào đang chạy (Cổng 5050 đã sạch sẵn).`n" -ForegroundColor Yellow
}
