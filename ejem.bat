@echo off
echo ========================================
echo   Chronos & LoRA - Setup Automático 🚀
echo ========================================

:: Paso 1 - Crear entorno virtual
echo.
echo Creando entorno virtual...
python -m venv myEnvPrueba

:: Paso 2 - Activar entorno
echo.
echo Activando entorno virtual...
call myEnvPrueba\Scripts\activate.bat

:: Paso 3 - Instalar dependencias de Chronos
echo.
echo Instalando dependencias del repositorio...
cd LoRA
pip install --editable ".[training]"

:: Paso 4 - Instalar PEFT
echo.
echo Instalando PEFT...
pip install peft

:: Paso 5 - Instalar PyTorch con soporte CUDA
echo.
echo Instalando PyTorch con soporte CUDA...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

:: Paso 6 - Ejecutar fine-tuning
echo.
echo Iniciando entrenamiento...
cd scripts
python training/train.py --config ./training/configs/chronos-t5-small-lora.yaml

pause
