# 🚀 Guida Rapida - Generatore di Cruciverba

## ⚡ Avvio Ultra-Rapido

```bash
# 1. Clona e avvia
git clone https://github.com/fabriziosalmi/crosswords.git
cd crosswords
make quick-start

# 2. Apri nel browser
open http://localhost:3000
```

## 🎯 Opzioni di Avvio

### 🏠 Locale (senza Docker)
```bash
npm install
npm start
# → http://localhost:3000
```

### 🐳 Docker - Sviluppo
```bash
make dev
# → http://localhost:3000
# ✅ Auto-reload attivo
```

### 🏭 Docker - Produzione
```bash
make prod
# → http://localhost (porta 80)
# ✅ Nginx + SSL + Rate limiting
```

### 🛠️ Docker - Con Strumenti
```bash
make dev-tools
# → Applicazione: http://localhost:3000
# → Portainer: http://localhost:9000
# → Adminer: http://localhost:8080
```

### 🐍 Solo Python
```bash
python3 simple_crossword.py
# Genera file HTML statico
```

## 🎮 Come Usare l'Interfaccia

### Controlli Principali
- **Genera Cruciverba**: Crea un nuovo puzzle
- **Verifica**: Controlla la soluzione
- **Suggerimento**: Rivela una lettera casuale
- **Pulisci**: Resetta la griglia
- **Stampa/Salva**: Esporta il cruciverba

### Navigazione Intelligente
- **Frecce** o **Tab**: Naviga tra le celle
- **Click su definizione**: Va alla prima cella della parola
- **Auto-focus**: Passa automaticamente alla cella successiva

### Impostazioni Personalizzabili
- **Dimensioni**: 10x10, 15x15, 20x20
- **Numero parole**: 8, 12, 16, 20
- **Temi**: Generale, Animali, Famiglia, Cibo, Natura

## 📊 Monitoraggio

### Health Check
```bash
curl http://localhost:3000/health
# oppure
make health
```

### API Info
```bash
curl http://localhost:3000/api/info
```

### Logs
```bash
make logs        # Solo app
make logs-all    # Tutti i servizi
```

### Status Servizi
```bash
make status      # Docker Compose
docker ps        # Tutti i container
```

## 🔧 Comandi Make Essenziali

| Comando | Descrizione | Quando Usarlo |
|---------|-------------|---------------|
| `make quick-start` | Avvio rapido | Prima volta |
| `make dev` | Sviluppo | Durante il coding |
| `make prod` | Produzione | Deploy finale |
| `make health` | Verifica salute | Debugging |
| `make logs` | Visualizza logs | Troubleshooting |
| `make clean` | Pulizia completa | Problemi Docker |
| `make shell` | Shell nel container | Debug avanzato |
| `make help` | Lista tutti i comandi | Quando sei perso |

## 🆘 Risoluzione Problemi Rapidi

### ❌ Porta già in uso
```bash
make down
# oppure cambia porta in docker-compose.yml
```

### ❌ Container non si avvia
```bash
make logs         # Vedi gli errori
make clean        # Pulisci tutto
make build        # Ricostruisci
make up           # Riavvia
```

### ❌ Problemi di memoria
```bash
make clean-all    # Libera spazio Docker
docker system prune -a
```

### ❌ L'interfaccia non funziona
- Controlla che il container sia "healthy": `make status`
- Verifica l'health check: `make health`
- Guarda i logs: `make logs`

## 🌐 Endpoint Utili

- **Applicazione**: http://localhost:3000
- **Health Check**: http://localhost:3000/health
- **API Info**: http://localhost:3000/api/info
- **File Generati**: http://localhost:3000/api/files

## 🔄 Aggiornamenti

```bash
git pull origin main
make update
# Scarica, ricostruisce e riavvia tutto
```

## 📱 Accesso da Altri Dispositivi

L'applicazione è disponibile sulla rete locale:
```bash
# Trova il tuo IP locale
hostname -I
# Poi visita: http://TUO_IP:3000
```

## 🎯 Pro Tips

1. **Usa `make dev` per sviluppo** - ha auto-reload
2. **`make prod` per testing finale** - ambiente identico alla produzione  
3. **`make health` è tuo amico** - usalo spesso per verifiche rapide
4. **I logs sono in tempo reale** - `make logs` ti dice tutto
5. **`make clean` risolve il 90% dei problemi Docker**

---

**💡 Non funziona qualcosa?**
1. `make status` - vedi lo stato
2. `make health` - verifica la salute  
3. `make logs` - guarda gli errori
4. `make clean && make build && make up` - reset completo

**🎉 Tutto funziona?** Inizia a creare cruciverba fantastici! 🧩
