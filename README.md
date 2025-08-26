# Generatore di Cruciverba Moderno

🧩 Un generatore di cruciverba in italiano con interfaccia moderna e funzionalità avanzate

✅ **COMPLETAMENTE RINNOVATO!** - Nuova interfaccia utente moderna con UX migliorata  
🐳 **DOCKER READY!** - Deploy facile con Docker e Docker Compose

## 🚀 Quick Start con Docker

```bash
# Clona il repository
git clone https://github.com/your-username/crosswords.git
cd crosswords

# Avvia con Docker Compose
make quick-start
# oppure: docker-compose up --build -d

# Apri nel browser
open http://localhost:3000
```

## ✨ Caratteristiche Principali

### 🎨 **Interfaccia Moderna**
- Design moderno con gradiente e animazioni
- Layout responsive per tutti i dispositivi
- Tema colori coerente e professionale
- Icone Font Awesome integrate
- Tipografia Google Fonts (Poppins)

### 🎮 **Funzionalità di Gioco**
- Timer integrato per tracciare il tempo di risoluzione
- Barra di progresso dinamica
- Sistema di suggerimenti intelligenti
- Verifica automatica delle soluzioni
- Modal di completamento con statistiche
- Messaggi di stato in tempo reale

### 🛠️ **Controlli Avanzati**
- Selezione dimensione griglia (10x10, 15x15, 20x20)
- Numero parole configurabile (8-20)
- Temi selezionabili (Generale, Animali, Famiglia, Cibo, Natura)
- Reset completo del gioco
- Esportazione HTML e stampa

### ⌨️ **Navigazione Intelligente**
- Navigazione con frecce direzionali
- Auto-focus sulla cella successiva
- Highlighting delle parole correlate
- Feedback visivo per risposte corrette/errate
- Supporto completo da tastiera

### 📱 **Responsive Design**
- Ottimizzato per desktop, tablet e mobile
- Layout adattivo con breakpoint multipli
- Controlli touch-friendly
- Interfaccia scalabile

### 🐳 **Docker Support**
- Containerizzazione completa
- Docker Compose per easy deployment
- Configurazioni separate per dev/prod
- Health checks automatici
- Nginx reverse proxy incluso

## 🚀 Come Utilizzare

### Opzione 1: Docker (Consigliata)

#### Development Mode
```bash
make dev
# Auto-reload abilitato, perfetto per sviluppo
```

#### Production Mode
```bash
make prod
# Con Nginx, SSL, rate limiting, health checks
```

#### Con Strumenti di Sviluppo
```bash
make dev-tools
# Include Portainer (Docker GUI) e Adminer
```

### Opzione 2: Locale
```bash
# Installa dipendenze
npm install

# Avvia server
npm start
# oppure: node server.js

# Sviluppo con auto-reload
npm run dev
```

### Opzione 3: Python Standalone
```bash
python3 simple_crossword.py
# Genera un file HTML statico
```

## 🛠️ Comandi Make Disponibili

```bash
make help                # Mostra tutti i comandi
make quick-start         # Avvio rapido per nuovi utenti
make dev                 # Ambiente di sviluppo
make prod               # Ambiente di produzione
make dev-tools          # Sviluppo + strumenti aggiuntivi
make build              # Costruisce l'applicazione
make up/down            # Avvia/ferma servizi
make logs               # Visualizza logs
make health             # Verifica salute applicazione
make clean              # Pulizia container e volumi
make shell              # Shell nel container
make backup             # Backup dati
make ssl                # Genera certificati SSL
make info               # Informazioni ambiente
```

## 📚 Dizionario Italiano Espanso

Oltre **200 parole** italiane con definizioni intelligenti:

| Categoria | Esempi | Numero Parole |
|-----------|---------|---------------|
| 🐱 **Animali** | gatto, leone, pesce, uccello | 25+ |
| 👨‍👩‍👧‍👦 **Famiglia** | madre, padre, figlio, nonno | 15+ |
| 👁️ **Corpo Umano** | testa, mano, occhio, cuore | 15+ |
| 🏠 **Casa** | tavolo, sedia, porta, cucina | 20+ |
| 🍕 **Cibo** | pane, pasta, pizza, formaggio | 20+ |
| 🌿 **Natura** | sole, mare, albero, fiore | 20+ |
| 🎨 **Colori** | rosso, blu, verde, giallo | 10+ |
| ⏰ **Tempo** | anno, oggi, mattina, sera | 15+ |
| 💭 **Emozioni** | amore, gioia, pace, vita | 10+ |
| ⚡ **Verbi** | essere, avere, fare, dire | 25+ |
| 📏 **Aggettivi** | grande, bello, nuovo, veloce | 20+ |
| 👨‍⚕️ **Professioni** | medico, maestro, cuoco, poeta | 15+ |

## 🏗️ Architettura del Sistema

### 🎨 **Frontend Moderno**
- **HTML5 + CSS3**: Interfaccia responsive con design professionale
- **JavaScript ES6+**: Logica di gioco avanzata e interazione utente
- **Font Awesome + Google Fonts**: Icone e tipografia moderne

### 🔧 **Backend Node.js**
- **Express Server**: API REST per gestione applicazione
- **Python Integration**: Generatore di cruciverba `simple_crossword.py`
- **Health Monitoring**: Endpoint di stato e debugging

### 🐳 **Containerizzazione Docker**
- **Multi-environment**: Development, staging, production
- **Nginx Proxy**: Load balancing e SSL termination
- **Health Checks**: Monitoraggio automatico dei servizi

## 🔧 API Endpoints

| Endpoint | Metodo | Descrizione |
|----------|--------|--------------|
| `/` | GET | Interfaccia principale |
| `/health` | GET | Status dell'applicazione |
| `/api/info` | GET | Informazioni API |
| `/api/files` | GET | Lista file generati |
| `/api/generate-python` | POST | Genera cruciverba con Python |

## 📋 Requisiti Sistema

### 🐳 **Con Docker (Raccomandato)**
- Docker 20.10+
- Docker Compose 2.0+
- 2GB RAM disponibile
- 1GB spazio disco

### 🖥️ **Installazione Locale**
- Node.js 16.0+
- Python 3.8+
- npm 8.0+

## Contributing

Contributions are welcome!  Here are some ways you can contribute:

*   **Report Bugs:**  If you encounter any issues, please open an issue on the GitHub repository, providing a detailed description of the problem, steps to reproduce it, and your system configuration (OS, Python version, etc.).
*   **Suggest Features:**  If you have ideas for new features or improvements, please open an issue and describe your suggestion.
*   **Submit Pull Requests:**  If you've implemented a bug fix or new feature, feel free to submit a pull request.  Please ensure your code follows the existing style and includes appropriate comments and documentation.
* **Improve Word Lists:** contribute by enhancing or creating word lists for different languages.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. (You'll need to create a LICENSE file and put the MIT license text in it.)

