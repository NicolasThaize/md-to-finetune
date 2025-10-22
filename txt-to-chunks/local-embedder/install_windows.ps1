# Requires -Version 5.1
# PowerShell script to install Ollama on Windows and pull the Mistral 7B Instruct model
# Run this script from an elevated PowerShell (Run as Administrator)

# Utility: Write info with emoji
function Write-Info($msg) { Write-Host $msg -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host $msg -ForegroundColor Green }
function Write-Warn($msg) { Write-Host $msg -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host $msg -ForegroundColor Red }

# Ensure TLS 1.2 for downloads
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# Check admin
function Test-IsAdmin {
  $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
  $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
  return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-IsAdmin)) {
  Write-Err "❌ Ce script doit être exécuté en tant qu'administrateur (PowerShell 'Run as Administrator')."
  Write-Host "Astuce: Clic droit sur PowerShell → 'Exécuter en tant qu'administrateur'"
  exit 1
}

Write-Info "🍏 Installation d'Ollama pour Windows"
Write-Host "==================================="

# Check if Ollama already installed
function Test-OllamaInstalled {
  try {
    $cmd = Get-Command ollama -ErrorAction Stop
    return $true
  } catch {
    return $false
  }
}

if (Test-OllamaInstalled) {
  Write-Ok "✅ Ollama déjà installé"
  try { ollama --version } catch { Write-Warn "(Impossible d'obtenir la version)" }
} else {
  Write-Info "📦 Installation via winget (recommandé)..."
  # Check winget
  $hasWinget = $false
  try {
    $null = Get-Command winget -ErrorAction Stop
    $hasWinget = $true
  } catch {
    $hasWinget = $false
  }

  if ($hasWinget) {
    try {
      winget install -e --id Ollama.Ollama --silent --accept-package-agreements --accept-source-agreements
      Write-Ok "✅ Ollama installé via winget"
    } catch {
      Write-Warn "⚠️  Echec de l'installation via winget: $($_.Exception.Message)"
      Write-Host "Veuillez installer manuellement depuis: https://ollama.ai/download"
      exit 1
    }
  } else {
    Write-Warn "⚠️  winget indisponible."
    Write-Host "Veuillez installer Ollama manuellement: https://ollama.ai/download"
    exit 1
  }
}

# Try to start Ollama service
Write-Info "🚀 Démarrage du service Ollama..."
try {
  $svc = Get-Service -Name "Ollama" -ErrorAction SilentlyContinue
  if ($null -ne $svc) {
    if ($svc.Status -ne 'Running') { Start-Service -Name "Ollama" }
    Write-Ok "✅ Service Windows 'Ollama' en cours d'exécution"
  } else {
    # Fallback to foreground serve
    Write-Warn "⚠️  Service Windows introuvable, lancement en mode utilisateur"
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
  }
} catch {
  Write-Warn "⚠️  Impossible de démarrer le service automatiquement: $($_.Exception.Message)"
}

# Wait for readiness
Write-Info "🔄 Attente que Ollama soit prêt..."
$ready = $false
for ($i = 1; $i -le 15; $i++) {
  try {
    $null = & ollama list 2>$null
    $ready = $true
    break
  } catch {
    Start-Sleep -Seconds 2
  }
}

if (-not $ready) {
  Write-Err "❌ Ollama ne semble pas prêt. Vérifiez que l'application Ollama est lancée."
  Write-Host "Ouvrez 'Ollama' depuis le menu Démarrer puis relancez ce script."
  exit 1
}

# Pull model
Write-Info "🤖 Téléchargement du modèle Mistral 7B Instruct (peut prendre du temps)..."
try {
  ollama pull mistral:7b-instruct
  Write-Ok "✅ Modèle téléchargé"
} catch {
  Write-Err "❌ Echec lors du téléchargement du modèle: $($_.Exception.Message)"
  exit 1
}

# Verify
Write-Info "🔍 Vérification..."
try {
  ollama list
  Write-Ok "✅ Installation terminée"
  Write-Host "\n💡 Pour tester:"
  Write-Host "   ollama run mistral:7b-instruct"
  Write-Host "\n💡 Pour générer le dataset (depuis WSL ou Python Windows):"
  Write-Host "   python dataset_generator.py"
} catch {
  Write-Warn "⚠️  Ollama list a échoué, mais l'installation peut être correcte."
}
