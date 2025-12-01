# GitHub Secrets Configuration Guide

Este documento describe los secretos necesarios para configurar en GitHub Actions para el despliegue automático.

## Secretos Requeridos

Configura estos secretos en: `Settings > Secrets and variables > Actions > New repository secret`

### 1. AZURE_CREDENTIALS
**Descripción:** Credenciales de Service Principal para autenticación con Azure CLI

**Cómo obtenerlo:**
```bash
az ad sp create-for-rbac --name "github-actions-gender-classifier" \
  --role contributor \
  --scopes /subscriptions/b9528b50-43fe-4d34-8855-b13f9febde9d/resourceGroups/rgmlopsicesiu2 \
  --sdk-auth
```

**Formato:** (JSON completo del comando anterior)
```json
{
  "clientId": "XXXX",
  "clientSecret": "XXXX",
  "subscriptionId": "b9528b50-43fe-4d34-8855-b13f9febde9d",
  "tenantId": "XXXX",
  "activeDirectoryEndpointUrl": "https://login.microsoftonline.com",
  "resourceManagerEndpointUrl": "https://management.azure.com/",
  "activeDirectoryGraphResourceId": "https://graph.windows.net/",
  "sqlManagementEndpointUrl": "https://management.core.windows.net:8443/",
  "galleryEndpointUrl": "https://gallery.azure.com/",
  "managementEndpointUrl": "https://management.core.windows.net/"
}
```

**Nota:** Copia todo el output JSON como está y pégalo en el secret de GitHub.

---

### 2. AZURE_STORAGE_CONNECTION_STRING
**Descripción:** Connection string para acceder a Azure Blob Storage

**Cómo obtenerlo:**
1. Ve a Azure Portal
2. Navega a: Storage Account `samlopsicesiu2`
3. En el menú lateral: `Security + networking` > `Access keys`
4. Copia el "Connection string" de key1 o key2

**Formato:**
```
DefaultEndpointsProtocol=https;AccountName=samlopsicesiu2;AccountKey=XXXX;EndpointSuffix=core.windows.net
```

---

### 3. AZURE_ACR_USERNAME
**Descripción:** Nombre de usuario para Azure Container Registry

**Cómo obtenerlo:**
1. Ve a Azure Portal
2. Navega a: Container Registry `acrmlopsicesiu2`
3. En el menú lateral: `Settings` > `Access keys`
4. Activa "Admin user" si no está activado
5. Copia el "Username"

**Valor típico:**
```
acrmlopsicesiu2
```

---

### 4. AZURE_ACR_PASSWORD
**Descripción:** Contraseña para Azure Container Registry

**Cómo obtenerlo:**
1. Mismo lugar que AZURE_ACR_USERNAME
2. Copia "password" o "password2"

**Formato:** (string de caracteres aleatorios)

---

## Verificación de Secretos

Una vez configurados todos los secretos, deberías tener en GitHub:

```
Settings > Secrets and variables > Actions > Repository secrets

✓ AZURE_CREDENTIALS
✓ AZURE_STORAGE_CONNECTION_STRING
✓ AZURE_ACR_USERNAME
✓ AZURE_ACR_PASSWORD
```

## Datos de Prueba en Azure

Antes de ejecutar el pipeline, asegúrate de que existan estos recursos en Azure Blob Storage:

### Container: `testdata`
- Archivo: `test_samples.npz` (formato NPZ para datos de imágenes)
- Contenido: Datos de prueba para validar el modelo de clasificación de género

### Container: `models`
- Archivo: `gender_googlenet.onnx`
- URL: https://samlopsicesiu2.blob.core.windows.net/models/gender_googlenet.onnx

### Container: `predictions`
- Este container se creará automáticamente
- Contendrá: `predicciones_dev.txt` y `predicciones_prod.txt`

## Configuración del Web App

### Crear Web App de Desarrollo (si no existe)

```bash
az webapp create \
  --resource-group rgmlopsicesiu2 \
  --plan <your-app-service-plan> \
  --name genderclassifiermodel-dev \
  --deployment-container-image-name acrmlopsicesiu2.azurecr.io/gender-classifier:dev-latest
```

### Configurar Web App para usar Container Registry

Para ambos Web Apps (dev y prod):

```bash
# Configurar para dev
az webapp config container set \
  --name genderclassifiermodel-dev \
  --resource-group rgmlopsicesiu2 \
  --container-image-name acrmlopsicesiu2.azurecr.io/gender-classifier:dev-latest \
  --container-registry-url https://acrmlopsicesiu2.azurecr.io \
  --container-registry-user <ACR_USERNAME> \
  --container-registry-password <ACR_PASSWORD>

# Configurar para prod
az webapp config container set \
  --name genderclassifiermodel \
  --resource-group rgmlopsicesiu2 \
  --container-image-name acrmlopsicesiu2.azurecr.io/gender-classifier:prod-latest \
  --container-registry-url https://acrmlopsicesiu2.azurecr.io \
  --container-registry-user <ACR_USERNAME> \
  --container-registry-password <ACR_PASSWORD>
```

**Nota:** Los comandos usan los nuevos parámetros `--container-*` en lugar de los deprecated `--docker-*`.

## Prueba del Pipeline

1. Haz un commit y push a la rama `dev`:
```bash
git add .
git commit -m "test: trigger dev pipeline"
git push origin dev
```

2. Ve a GitHub Actions para ver el progreso
3. Verifica que ambas etapas (test y build-and-deploy) se ejecuten correctamente

4. Una vez exitoso en dev, haz merge a `prod`:
```bash
git checkout prod
git merge dev
git push origin prod
```
