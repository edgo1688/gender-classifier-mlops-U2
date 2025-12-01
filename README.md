# Gender Classifier MLOps - CI/CD Pipeline

Sistema de despliegue automático para modelo ONNX de clasificación de género con pipeline CI/CD usando GitHub Actions y Azure.

## 📋 Tabla de Contenidos

- [Arquitectura](#arquitectura)
- [Características](#características)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Configuración Inicial](#configuración-inicial)
- [Pipeline CI/CD](#pipeline-cicd)
- [Endpoints](#endpoints)
- [Uso de la API](#uso-de-la-api)
- [Desarrollo Local](#desarrollo-local)
- [Monitoreo](#monitoreo)

## 🏗️ Arquitectura

```
┌─────────────────┐
│   GitHub Repo   │
│   (dev/prod)    │
└────────┬────────┘
         │
         │ Push trigger
         ▼
┌─────────────────────────────────────┐
│      GitHub Actions Pipeline        │
│  ┌─────────────────────────────┐   │
│  │  Stage 1: Test              │   │
│  │  - Download test data       │   │
│  │  - Download ONNX model      │   │
│  │  - Run unit tests           │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │  Stage 2: Build & Deploy    │   │
│  │  - Build Docker image       │   │
│  │  - Push to ACR              │   │
│  │  - Deploy to Azure Web App  │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Azure Container Registry (ACR)    │
│   acrmlopsicesiu2.azurecr.io        │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│      Azure Web Apps                 │
│  ┌─────────────────────────────┐   │
│  │  DEV Environment            │   │
│  │  genderclassifiermodel-dev  │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │  PROD Environment           │   │
│  │  genderclassifiermodel      │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
         │
         │ Logs predictions
         ▼
┌─────────────────────────────────────┐
│   Azure Blob Storage                │
│   samlopsicesiu2                    │
│  - predicciones_dev.txt             │
│  - predicciones_prod.txt            │
└─────────────────────────────────────┘
```

## ✨ Características

### Pipeline Automatizado
- ✅ **Testing automático** al hacer push a `dev` o `prod`
- ✅ **Descarga dinámica** de modelo ONNX desde Azure Blob Storage
- ✅ **Pruebas unitarias** con validación de métricas
- ✅ **Build de contenedor Docker** automático
- ✅ **Despliegue a Azure Web App** con health checks

### API REST con FastAPI
- ✅ Endpoints para predicciones
- ✅ Health check integrado
- ✅ Información del modelo
- ✅ Logging automático de predicciones

### Monitoreo
- ✅ **Registro de predicciones** en Azure Blob Storage
- ✅ Archivo separado para dev (`predicciones_dev.txt`) y prod (`predicciones_prod.txt`)
- ✅ Formato: timestamp, ambiente, input y predicción

## 📁 Estructura del Proyecto

```
gender-classifier-mlops-U2/
├── .github/
│   └── workflows/
│       ├── dev.yml              # Pipeline CI/CD para dev
│       └── prod.yml             # Pipeline CI/CD para prod
├── src/
│   └── app.py                   # Aplicación FastAPI
├── tests/
│   └── test_model.py            # Tests unitarios del modelo
├── Dockerfile                   # Configuración Docker
├── requirements.txt             # Dependencias Python
├── .env.example                 # Template de variables de entorno
├── .gitignore                   # Archivos ignorados por Git
├── SECRETS_SETUP.md             # Guía de configuración de secretos
└── README.md                    # Este archivo
```

## 🚀 Configuración Inicial

### 1. Clonar el Repositorio

```bash
git clone https://github.com/edgo1688/gender-classifier-mlops-U2.git
cd gender-classifier-mlops-U2
```

### 2. Configurar Secretos de GitHub

Sigue la guía detallada en [SECRETS_SETUP.md](SECRETS_SETUP.md) para configurar:

- `AZURE_CREDENTIALS`
- `AZURE_STORAGE_CONNECTION_STRING`
- `AZURE_ACR_USERNAME`
- `AZURE_ACR_PASSWORD`

### 3. Preparar Azure Blob Storage

Asegúrate de tener estos containers en `samlopsicesiu2`:

#### Container: `models`
Contiene el modelo ONNX:
- `gender_googlenet.onnx`
- URL: https://samlopsicesiu2.blob.core.windows.net/models/gender_googlenet.onnx

#### Container: `testdata`
Contiene datos de prueba:
- `test_samples.npz` - Datos de imágenes para validación del modelo

#### Container: `predictions`
Se crea automáticamente para almacenar logs:
- `predicciones_dev.txt`
- `predicciones_prod.txt`

### 4. Crear Web App de Desarrollo (Opcional)

Si no existe `genderclassifiermodel-dev`:

```bash
az webapp create \
  --resource-group rgmlopsicesiu2 \
  --plan <your-app-service-plan> \
  --name genderclassifiermodel-dev \
  --deployment-container-image-name acrmlopsicesiu2.azurecr.io/gender-classifier:dev-latest
```

## 🔄 Pipeline CI/CD

### Flujo de Trabajo

1. **Desarrollo en `dev`:**
   ```bash
   git checkout dev
   # Hacer cambios
   git add .
   git commit -m "feat: nueva funcionalidad"
   git push origin dev
   ```

2. **Pipeline se ejecuta automáticamente:**
   - **Stage: Test**
     - Descarga datos de prueba desde Azure
     - Descarga modelo ONNX
     - Ejecuta tests unitarios
   
   - **Stage: Build & Deploy**
     - Construye imagen Docker
     - Push a Azure Container Registry
     - Despliega a Web App dev

3. **Promoción a producción:**
   ```bash
   git checkout prod
   git merge dev
   git push origin prod
   ```

### Tests Implementados

1. **test_model_responds_with_defined_input**
   - Verifica que el modelo responde con datos de entrada definidos
   - Valida que las predicciones no sean nulas

2. **test_model_performance_metric**
   - Calcula la accuracy del modelo con datos de prueba
   - Valida que la métrica supera el umbral definido (50%)
   - Threshold configurable en `tests/test_model.py`

## 🌐 Endpoints

### Desarrollo
- **Base URL:** `https://genderclassifiermodel-dev.azurewebsites.net`

### Producción
- **Base URL:** `https://genderclassifiermodel.azurewebsites.net`

### Endpoints Disponibles

#### `GET /`
Interfaz web para clasificación de género
- Interfaz de usuario para subir imágenes
- Predicción en tiempo real con resultados visuales

#### `GET /api`
Información básica de la API
```json
{
  "message": "Gender Classifier MLOps API",
  "environment": "dev|prod",
  "status": "running",
  "model": "gender_googlenet"
}
```

#### `GET /health`
Health check del servicio
```json
{
  "status": "healthy",
  "environment": "dev|prod",
  "model_loaded": true
}
```

#### `POST /predict/image` (Recomendado)
Realizar predicción desde imagen

**Request:** Multipart form data
```bash
curl -X POST https://genderclassifiermodel-dev.azurewebsites.net/predict/image \
  -F "file=@path/to/face_image.jpg"
```

**Response:**
```json
{
  "gender": "male",
  "confidence": 0.87,
  "probabilities": {
    "female": 0.13,
    "male": 0.87
  },
  "environment": "dev",
  "timestamp": "2025-11-30T10:30:00.000000"
}
```

#### `POST /predict` (Avanzado)
Realizar predicción con tensor preprocesado

**Request:**
```json
{
  "input_data": [[...150528 float values...]]
}
```

**Nota:** Este endpoint requiere tensores preprocesados en formato [1, 150528] o [1, 3, 224, 224]. Para uso regular, utilizar `/predict/image`.

**Response:**
```json
{
  "predictions": [[0.13, 0.87]],
  "environment": "dev",
  "timestamp": "2025-11-30T10:30:00.000000"
}
```

#### `GET /model/info`
Información del modelo
```json
{
  "inputs": [
    {
      "name": "input",
      "shape": [1, 3, 224, 224],
      "type": "tensor(float)"
    }
  ],
  "outputs": [
    {
      "name": "loss3/loss3_Y",
      "shape": [1, 2],
      "type": "tensor(float)"
    }
  ],
  "environment": "dev",
  "model_type": "gender_classification",
  "description": "GenderNet (GoogLeNet-based) - Classifies gender from facial images"
}
```

## 💻 Uso de la API

### Con cURL

```bash
# Health check
curl https://genderclassifiermodel-dev.azurewebsites.net/health

# Predicción con imagen (recomendado)
curl -X POST https://genderclassifiermodel-dev.azurewebsites.net/predict/image \
  -F "file=@path/to/face_image.jpg"
```

### Con Python

```python
import requests

# Endpoint para imágenes
url = "https://genderclassifiermodel-dev.azurewebsites.net/predict/image"

# Subir imagen
with open('path/to/face_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)

result = response.json()
print(f"Gender: {result['gender']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")
```

### Con JavaScript

```javascript
const url = 'https://genderclassifiermodel-dev.azurewebsites.net/predict/image';

// Desde un input file
const fileInput = document.getElementById('imageInput');
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch(url, {
  method: 'POST',
  body: formData,
})
  .then(response => response.json())
  .then(data => {
    console.log('Gender:', data.gender);
    console.log('Confidence:', data.confidence);
    console.log('Probabilities:', data.probabilities);
  });
```

## 🛠️ Desarrollo Local

### Configuración del Entorno

1. **Crear entorno virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurar variables de entorno:**
   ```bash
   cp .env.example .env
   # Editar .env con tus credenciales
   ```

4. **Descargar modelo:**
   ```bash
   mkdir -p model
   curl -L -o model/gender_googlenet.onnx \
     https://samlopsicesiu2.blob.core.windows.net/models/gender_googlenet.onnx
   ```

### Ejecutar Aplicación Local

```bash
# Ejecutar con uvicorn
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000

# O ejecutar directamente
python src/app.py
```

Acceder a: http://localhost:8000

### Ejecutar Tests

```bash
# Todos los tests
pytest tests/ -v

# Tests específicos
pytest tests/test_model.py::test_model_responds_with_defined_input -v

# Con coverage
pytest tests/ --cov=src --cov-report=html
```

### Build Docker Local

```bash
# Build
docker build -t gender-classifier:local .

# Run
docker run -p 8000:8000 \
  -e ENVIRONMENT=dev \
  -e AZURE_STORAGE_CONNECTION_STRING="<your-connection-string>" \
  gender-classifier:local

# Acceder a http://localhost:8000
```

## 📊 Monitoreo

### Logs de Predicciones

Todas las predicciones se registran automáticamente en Azure Blob Storage:

- **Container:** `predictions`
- **Archivos:**
  - `predicciones_dev.txt` - Predicciones del entorno de desarrollo
  - `predicciones_prod.txt` - Predicciones del entorno de producción

**Formato de log:**
```
2025-11-29T10:30:00.000000 | ENV: dev | INPUT: [[1.0, 2.0, ...]] | PREDICTION: [[0.85, 0.15]]
```

### Visualizar Logs

#### Desde Azure Portal
1. Ve a Storage Account `samlopsicesiu2`
2. Navega a Containers > `predictions`
3. Descarga `predicciones_dev.txt` o `predicciones_prod.txt`

#### Con Azure CLI
```bash
# Ver logs de dev
az storage blob download \
  --account-name samlopsicesiu2 \
  --container-name predictions \
  --name predicciones_dev.txt \
  --file predicciones_dev.txt

# Ver logs de prod
az storage blob download \
  --account-name samlopsicesiu2 \
  --container-name predictions \
  --name predicciones_prod.txt \
  --file predicciones_prod.txt
```

#### Con Python
```python
from azure.storage.blob import BlobServiceClient

connection_string = "your-connection-string"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Leer logs de dev
blob_client = blob_service_client.get_blob_client(
    container="predictions",
    blob="predicciones_dev.txt"
)
logs = blob_client.download_blob().readall().decode('utf-8')
print(logs)
```

### Análisis de Logs

Puedes analizar los logs para:
- **Monitoreo de uso:** Cantidad de predicciones por día/hora
- **Análisis de inputs:** Distribución de características de entrada
- **Drift detection:** Comparar distribuciones de predicciones en el tiempo
- **Debugging:** Revisar predicciones específicas

## 🔧 Troubleshooting

### Pipeline falla en tests
- Verifica que los datos de prueba existan en Azure Blob Storage
- Revisa los logs del job en GitHub Actions
- Asegúrate de que el modelo se descargue correctamente

### Deployment falla
- Verifica que los secretos estén configurados correctamente
- Revisa que el Azure Container Registry sea accesible
- Confirma que el Web App existe y está configurado

### Logs no se guardan
- Verifica `AZURE_STORAGE_CONNECTION_STRING` en Web App settings
- Confirma que el container `predictions` existe
- Revisa los logs de la aplicación en Azure Portal

## 📝 Variables de Entorno

### Aplicación
- `ENVIRONMENT`: Ambiente actual (`dev` o `prod`)
- `MODEL_URL`: URL del modelo ONNX en Azure Blob Storage
- `MODEL_PATH`: Ruta local del modelo
- `AZURE_STORAGE_CONNECTION_STRING`: Connection string para Azure Storage
- `PREDICTIONS_CONTAINER`: Container para guardar logs de predicciones

### Pipeline (GitHub Secrets)
- `AZURE_CREDENTIALS` - Credenciales de Service Principal
- `AZURE_STORAGE_CONNECTION_STRING`
- `AZURE_ACR_USERNAME`
- `AZURE_ACR_PASSWORD`

## 🤝 Contribuir

1. Fork el repositorio
2. Crea una rama feature: `git checkout -b feature/nueva-funcionalidad`
3. Commit tus cambios: `git commit -m 'feat: nueva funcionalidad'`
4. Push a la rama: `git push origin feature/nueva-funcionalidad`
5. Abre un Pull Request hacia `dev`

## 📄 Licencia

Ver archivo [LICENSE](LICENSE)

## 👥 Autor

- Edwin Gómez
- Machine learning operations – MLOps
- Universidad ICESI

## 🔗 Enlaces Útiles

- [Azure Portal](https://portal.azure.com)
- [Container Registry](https://portal.azure.com/#@edgo1688hotmail.onmicrosoft.com/resource/subscriptions/b9528b50-43fe-4d34-8855-b13f9febde9d/resourcegroups/rgmlopsicesiu2/providers/Microsoft.ContainerRegistry/registries/acrmlopsicesiu2/overview)
- [Storage Account](https://portal.azure.com/#@edgo1688hotmail.onmicrosoft.com/resource/subscriptions/b9528b50-43fe-4d34-8855-b13f9febde9d/resourcegroups/rgmlopsicesiu2/providers/Microsoft.Storage/storageAccounts/samlopsicesiu2/storagebrowser)
- [Web App Prod](https://portal.azure.com/#@edgo1688hotmail.onmicrosoft.com/resource/subscriptions/b9528b50-43fe-4d34-8855-b13f9febde9d/resourcegroups/rgmlopsicesiu2/providers/Microsoft.Web/sites/genderclassifiermodel/appServices)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ONNX Runtime](https://onnxruntime.ai/)
