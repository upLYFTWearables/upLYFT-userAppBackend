# Portable Node.js Setup Script for React Dashboard
# This downloads Node.js temporarily without system installation

Write-Host "Setting up portable Node.js..." -ForegroundColor Green

# Create temp directory
$nodeDir = ".\node-portable"
$nodeZip = "node-v20.15.1-win-x64.zip"
$nodeUrl = "https://nodejs.org/dist/v20.15.1/$nodeZip"

# Download Node.js if not exists
if (!(Test-Path $nodeDir)) {
    Write-Host "Downloading Node.js..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $nodeUrl -OutFile $nodeZip
    
    Write-Host "Extracting Node.js..." -ForegroundColor Yellow
    Expand-Archive -Path $nodeZip -DestinationPath "."
    Rename-Item "node-v20.15.1-win-x64" $nodeDir
    Remove-Item $nodeZip
}

# Add Node.js to PATH for this session
$env:PATH = "$PWD\$nodeDir;$env:PATH"

Write-Host "Node.js ready! Installing React dependencies..." -ForegroundColor Green

# Install dependencies and start
& "$nodeDir\npm.cmd" install
& "$nodeDir\npm.cmd" start

Write-Host "React app started! Check http://localhost:3000" -ForegroundColor Cyan 