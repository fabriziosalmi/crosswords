# 🐳 Docker Deployment Guide

Guida completa per l'implementazione e distribuzione del Generatore di Cruciverba usando Docker.

## 🚀 Quick Start

### Prerequisiti
- Docker Desktop (Windows/Mac) o Docker Engine (Linux)
- Docker Compose 2.0+
- Git (per clonare il repository)

### Avvio Rapido

```bash
# 1. Clona il repository
git clone https://github.com/your-username/crosswords.git
cd crosswords

# 2. Avvia l'applicazione
make quick-start

# 3. Apri nel browser
open http://localhost:3000
```

## 📦 Opzioni di Deployment

### 1. Development Mode
Per sviluppo locale con hot-reload:

```bash
make dev
```

Servizi disponibili:
- 🌐 **Applicazione**: http://localhost:3000
- 🔄 Auto-reload abilitato per modifiche al codice

### 2. Development con Strumenti
Include strumenti di sviluppo aggiuntivi:

```bash
make dev-tools
```

Servizi disponibili:
- 🌐 **Applicazione**: http://localhost:3000
- 🐳 **Portainer** (Docker GUI): http://localhost:9000
- 🗄️ **Adminer** (DB Admin): http://localhost:8080

### 3. Production Mode
Per deployment in produzione con Nginx:

```bash
make prod
```

Servizi disponibili:
- 🌐 **Applicazione**: http://localhost (porta 80)
- 🔒 **HTTPS**: http://localhost:443 (con certificati SSL)
- 🚦 Load balancing e rate limiting
- 📊 Health checks automatici

## 🛠️ Comandi Make Disponibili

```bash
make help                # Mostra tutti i comandi disponibili
make dev                 # Avvia ambiente di sviluppo
make prod               # Avvia ambiente di produzione
make build              # Costruisce l'applicazione
make up                 # Avvia i servizi
make down               # Ferma i servizi
make restart            # Riavvia i servizi
make logs               # Mostra i logs dell'applicazione
make logs-all           # Mostra i logs di tutti i servizi
make status             # Mostra lo stato dei servizi
make health             # Verifica la salute dell'applicazione
make clean              # Pulizia container e volumi
make clean-all          # Pulizia completa incluse immagini
make shell              # Apre shell nel container
make backup             # Backup dei dati
make update             # Aggiorna e riavvia servizi
make info               # Informazioni ambiente
```

## 🏗️ Struttura Docker

### Dockerfile
- **Base**: `node:18-alpine` (leggero e sicuro)
- **Python**: Installato per il generatore Python
- **Express**: Server HTTP per servire l'applicazione
- **Health Check**: Endpoint `/health` per monitoraggio
- **Multi-stage**: Ottimizzato per produzione

### docker-compose.yml
- **crossword-generator**: Servizio principale dell'applicazione
- **nginx**: Reverse proxy per produzione (profilo `production`)
- **redis**: Cache opzionale (profilo `with-redis`)

### Configurazioni di Rete
- **Rete personalizzata**: `crossword-network`
- **Load balancing**: Tramite Nginx upstream
- **Health checks**: Automatici ogni 30s
- **Restart policy**: `unless-stopped`

## 🔧 Configurazione Avanzata

### Variabili d'Ambiente

| Variabile | Descrizione | Default |
|-----------|-------------|---------|
| `NODE_ENV` | Ambiente di esecuzione | `production` |
| `PORT` | Porta dell'applicazione | `3000` |

### Volumi Persistenti

```yaml
volumes:
  redis_data:           # Cache Redis (opzionale)
  - ./ssl:/etc/ssl/certs # Certificati SSL
  - .:/app               # Mount per sviluppo
```

### Profili Docker Compose

- **default**: Solo servizio principale
- **production**: Con Nginx reverse proxy
- **with-redis**: Include cache Redis
- **dev-tools**: Strumenti di sviluppo

## 🔒 Sicurezza

### Headers di Sicurezza (Nginx)
- `X-Frame-Options: SAMEORIGIN`
- `X-Content-Type-Options: nosniff`
- `X-XSS-Protection: 1; mode=block`
- `Content-Security-Policy` configurato
- `Referrer-Policy: strict-origin-when-cross-origin`

### Rate Limiting
- **API**: 10 richieste/secondo
- **Generale**: 30 richieste/secondo
- **Burst**: Gestito automaticamente

### SSL/TLS
```bash
# Genera certificati auto-firmati per sviluppo
make ssl

# Per produzione, sostituisci con certificati reali
# in ./ssl/cert.pem e ./ssl/key.pem
```

## 📊 Monitoring e Logging

### Health Checks
```bash
# Verifica manuale
curl http://localhost:3000/health

# Tramite Make
make health
```

### Log Management
```bash
# Logs dell'applicazione
make logs

# Logs di tutti i servizi
make logs-all

# Logs in tempo reale
docker-compose logs -f --tail=50
```

### Metriche Container
```bash
# Stato dei container
make status

# Uso risorse
docker stats

# Con Portainer (make dev-tools)
# Vai su http://localhost:9000
```

## 🚀 Deployment in Produzione

### 1. Preparazione Server
```bash
# Installa Docker e Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 2. Configurazione SSL
```bash
# Usa Let's Encrypt per certificati gratuiti
sudo apt install certbot
sudo certbot certonly --standalone -d yourdomain.com

# Copia certificati nella directory ssl/
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ssl/cert.pem
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ssl/key.pem
```

### 3. Deploy
```bash
# Clone repository
git clone https://github.com/your-username/crosswords.git
cd crosswords

# Configura variabili ambiente
cp .env.example .env
# Modifica .env con le tue configurazioni

# Avvia in produzione
make prod
```

### 4. Backup Automatico
```bash
# Aggiungi a crontab per backup giornalieri
0 2 * * * cd /path/to/crosswords && make backup

# Backup manuale
make backup
```

## 🔧 Troubleshooting

### Problemi Comuni

#### Container non si avvia
```bash
# Controlla logs
make logs

# Ricostruisci container
make clean
make build
make up
```

#### Porta già in uso
```bash
# Cambia porta nel docker-compose.yml
ports:
  - "3001:3000"  # Invece di 3000:3000
```

#### Problemi di permessi
```bash
# Linux: aggiungi user al gruppo docker
sudo usermod -aG docker $USER
# Logout e login di nuovo
```

#### Out of memory
```bash
# Aumenta memoria Docker Desktop
# Settings > Resources > Memory > 4GB+

# O aggiungi limite nei servizi
deploy:
  resources:
    limits:
      memory: 512M
```

### Debug Avanzato

```bash
# Entra nel container
make shell

# Verifica configurazione
docker-compose config

# Analizza immagini
make docker-images

# Controlla network
docker network ls
docker network inspect crossword-network
```

## 📈 Ottimizzazione Performance

### 1. Multi-stage Build
Il Dockerfile usa multi-stage per ridurre le dimensioni:
- Build stage: Include tutti i tool di build
- Runtime stage: Solo i file necessari

### 2. Caching
- **Docker layers**: Ottimizzato per riutilizzo layer
- **Redis**: Cache opzionale per dati applicazione
- **Nginx**: Cache statica per asset

### 3. Resource Limits
```yaml
# docker-compose.yml
deploy:
  resources:
    limits:
      cpus: '0.5'
      memory: 512M
    reservations:
      cpus: '0.25'
      memory: 256M
```

## 🌐 Scalabilità

### Load Balancing
```yaml
# docker-compose.yml
services:
  crossword-generator:
    deploy:
      replicas: 3
  
  nginx:
    depends_on:
      - crossword-generator
```

### Database Cluster
```yaml
# Per future espansioni con database
services:
  postgres-master:
    image: postgres:15
  
  postgres-replica:
    image: postgres:15
    depends_on:
      - postgres-master
```

## 📝 Configurazione CI/CD

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to server
        run: |
          ssh user@server "cd /app && git pull && make update"
```

### Docker Hub Auto-build
```bash
# Tag per auto-build
git tag -a v1.0.0 -m "Release 1.0.0"
git push origin v1.0.0
```

## 📚 Risorse Aggiuntive

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [Nginx Configuration](https://nginx.org/en/docs/)
- [Let's Encrypt](https://letsencrypt.org/)
- [Portainer Documentation](https://documentation.portainer.io/)
