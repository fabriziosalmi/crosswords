class CrosswordGenerator {
    constructor() {
        this.words = [];
        this.grid = [];
        this.placedWords = [];
        this.gridSize = 15;
        this.maxAttempts = 1000;
        this.usedWords = new Set();
        
        this.loadWords();
        this.initializeEventListeners();
    }

    async loadWords() {
        try {
            const response = await fetch('data/parole.txt');
            const text = await response.text();
            this.words = text.split('\n')
                .map(word => word.trim().toUpperCase())
                .filter(word => word.length >= 3 && word.length <= 12)
                .filter(word => /^[A-Z]+$/.test(word));
            
            console.log(`Loaded ${this.words.length} words`);
        } catch (error) {
            console.error('Error loading words:', error);
            // Fallback words for testing
            this.words = ['CASA', 'SOLE', 'MARE', 'VITA', 'AMORE', 'TEMPO', 'GIORNO', 'NOTTE'];
        }
    }

    initializeEventListeners() {
        document.getElementById('generateBtn').addEventListener('click', () => {
            this.generateCrossword();
        });

        document.getElementById('downloadBtn').addEventListener('click', () => {
            this.downloadPDF();
        });
    }

    initializeGrid() {
        this.grid = Array(this.gridSize).fill(null).map(() => 
            Array(this.gridSize).fill('')
        );
        this.placedWords = [];
    }

    generateCrossword() {
        console.log('Starting crossword generation...');
        
        this.initializeGrid();
        this.usedWords.clear();
        
        // Select words for the crossword (8-12 words)
        let selectedWords = this.selectWords(20); // Get more words to choose from
        console.log('Available words:', selectedWords.length);

        if (selectedWords.length === 0) {
            alert('No words available for crossword generation');
            return;
        }

        // Place the first word horizontally in the center
        const firstWord = selectedWords[0];
        const startRow = Math.floor(this.gridSize / 2);
        const startCol = Math.floor((this.gridSize - firstWord.length) / 2);
        
        if (this.placeWord(firstWord, startRow, startCol, 'horizontal')) {
            console.log('Placed first word:', firstWord);
        } else {
            alert('Could not place first word');
            return;
        }

        // Try to place remaining words with improved algorithm
        let wordsPlaced = 1;
        let totalAttempts = 0;
        const maxTotalAttempts = selectedWords.length * 50;
        
        while (wordsPlaced < 10 && totalAttempts < maxTotalAttempts) {
            // Get fresh list of available words
            const availableWords = selectedWords.filter(word => !this.usedWords.has(word));
            
            if (availableWords.length === 0) {
                break;
            }
            
            // Try each available word
            let wordPlaced = false;
            
            for (const word of availableWords) {
                const placements = this.findAllValidPlacements(word);
                
                if (placements.length > 0) {
                    // Choose a random valid placement
                    const placement = placements[Math.floor(Math.random() * placements.length)];
                    
                    if (this.placeWord(word, placement.row, placement.col, placement.direction)) {
                        console.log(`Placed word ${wordsPlaced + 1}: ${word} at (${placement.row}, ${placement.col}) ${placement.direction}`);
                        wordsPlaced++;
                        wordPlaced = true;
                        break;
                    }
                }
            }
            
            if (!wordPlaced) {
                totalAttempts++;
            }
        }

        console.log(`Successfully placed ${wordsPlaced} words`);
        this.renderCrossword();
        this.generateClues();
    }

    selectWords(count) {
        // Filter out already used words and sort by length and frequency considerations
        const availableWords = this.words.filter(word => !this.usedWords.has(word));
        
        const sortedWords = availableWords.sort((a, b) => {
            // Prefer words of medium length (4-8 characters)
            const aScore = this.getWordScore(a);
            const bScore = this.getWordScore(b);
            return bScore - aScore;
        });

        return sortedWords.slice(0, count);
    }

    getWordScore(word) {
        // Score based on length (prefer 4-8 chars) and common letters
        let score = 0;
        
        if (word.length >= 4 && word.length <= 8) {
            score += 10;
        } else if (word.length >= 3 && word.length <= 10) {
            score += 5;
        }

        // Bonus for common vowels and consonants that help with intersections
        const commonLetters = 'AEIOURNSTL';
        for (let char of word) {
            if (commonLetters.includes(char)) {
                score += 1;
            }
        }

        return score;
    }

    findAllValidPlacements(word) {
        const placements = [];

        // Find all possible intersection points with existing words
        for (let placedWord of this.placedWords) {
            for (let i = 0; i < word.length; i++) {
                for (let j = 0; j < placedWord.word.length; j++) {
                    if (word[i] === placedWord.word[j]) {
                        // Calculate position for perpendicular placement
                        if (placedWord.direction === 'horizontal') {
                            // Place new word vertically
                            const newRow = placedWord.row - i;
                            const newCol = placedWord.col + j;
                            
                            if (this.canPlaceWord(word, newRow, newCol, 'vertical')) {
                                placements.push({ row: newRow, col: newCol, direction: 'vertical' });
                            }
                        } else {
                            // Place new word horizontally
                            const newRow = placedWord.row + j;
                            const newCol = placedWord.col - i;
                            
                            if (this.canPlaceWord(word, newRow, newCol, 'horizontal')) {
                                placements.push({ row: newRow, col: newCol, direction: 'horizontal' });
                            }
                        }
                    }
                }
            }
        }

        return placements;
    }

    canPlaceWord(word, row, col, direction) {
        // Check bounds
        if (direction === 'horizontal') {
            if (row < 0 || row >= this.gridSize || col < 0 || col + word.length > this.gridSize) {
                return false;
            }
        } else {
            if (col < 0 || col >= this.gridSize || row < 0 || row + word.length > this.gridSize) {
                return false;
            }
        }

        // Check for conflicts
        for (let i = 0; i < word.length; i++) {
            const currentRow = direction === 'horizontal' ? row : row + i;
            const currentCol = direction === 'horizontal' ? col + i : col;
            const currentCell = this.grid[currentRow][currentCol];

            if (currentCell !== '' && currentCell !== word[i]) {
                return false; // Conflict
            }
        }

        // Check for proper word separation (no adjacent words)
        if (direction === 'horizontal') {
            // Check before word
            if (col > 0 && this.grid[row][col - 1] !== '') {
                return false;
            }
            // Check after word
            if (col + word.length < this.gridSize && this.grid[row][col + word.length] !== '') {
                return false;
            }
            
            // Check cells above and below each letter (except at valid intersections)
            for (let i = 0; i < word.length; i++) {
                const checkRow = row;
                const checkCol = col + i;
                const currentCell = this.grid[checkRow][checkCol];
                
                // If this is an empty cell, check for adjacent words
                if (currentCell === '') {
                    // Check above
                    if (checkRow > 0 && this.grid[checkRow - 1][checkCol] !== '') {
                        return false;
                    }
                    // Check below
                    if (checkRow < this.gridSize - 1 && this.grid[checkRow + 1][checkCol] !== '') {
                        return false;
                    }
                }
            }
        } else { // vertical
            // Check before word
            if (row > 0 && this.grid[row - 1][col] !== '') {
                return false;
            }
            // Check after word
            if (row + word.length < this.gridSize && this.grid[row + word.length][col] !== '') {
                return false;
            }
            
            // Check cells left and right of each letter (except at valid intersections)
            for (let i = 0; i < word.length; i++) {
                const checkRow = row + i;
                const checkCol = col;
                const currentCell = this.grid[checkRow][checkCol];
                
                // If this is an empty cell, check for adjacent words
                if (currentCell === '') {
                    // Check left
                    if (checkCol > 0 && this.grid[checkRow][checkCol - 1] !== '') {
                        return false;
                    }
                    // Check right
                    if (checkCol < this.gridSize - 1 && this.grid[checkRow][checkCol + 1] !== '') {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    getAdjacentCells(row, col, direction) {
        const adjacent = [];
        if (direction === 'horizontal') {
            // Check above and below
            if (row > 0) adjacent.push({ row: row - 1, col });
            if (row < this.gridSize - 1) adjacent.push({ row: row + 1, col });
        } else {
            // Check left and right
            if (col > 0) adjacent.push({ row, col: col - 1 });
            if (col < this.gridSize - 1) adjacent.push({ row, col: col + 1 });
        }
        return adjacent;
    }

    placeWord(word, row, col, direction) {
        // Double-check if we can place the word
        if (!this.canPlaceWord(word, row, col, direction)) {
            return false;
        }
        
        for (let i = 0; i < word.length; i++) {
            const currentRow = direction === 'horizontal' ? row : row + i;
            const currentCol = direction === 'horizontal' ? col + i : col;
            this.grid[currentRow][currentCol] = word[i];
        }

        this.placedWords.push({
            word,
            row,
            col,
            direction,
            number: this.placedWords.length + 1
        });
        
        this.usedWords.add(word);
        return true;
    }

    renderCrossword() {
        const container = document.getElementById('crosswordContainer');
        const table = document.createElement('table');
        table.className = 'crossword-grid';

        // Number the starting positions
        const numberedCells = this.getNumberedCells();

        for (let i = 0; i < this.gridSize; i++) {
            const row = document.createElement('tr');
            
            for (let j = 0; j < this.gridSize; j++) {
                const cell = document.createElement('td');
                
                if (this.grid[i][j] !== '') {
                    cell.className = 'crossword-cell';
                    
                    // Add number if this is the start of a word
                    const cellNumber = numberedCells[`${i}-${j}`];
                    if (cellNumber) {
                        const numberSpan = document.createElement('span');
                        numberSpan.className = 'cell-number';
                        numberSpan.textContent = cellNumber;
                        cell.appendChild(numberSpan);
                    }

                    const input = document.createElement('input');
                    input.type = 'text';
                    input.maxLength = 1;
                    input.dataset.answer = this.grid[i][j];
                    cell.appendChild(input);
                } else {
                    cell.className = 'empty-cell';
                }
                
                row.appendChild(cell);
            }
            
            table.appendChild(row);
        }

        container.innerHTML = '';
        container.appendChild(table);
    }

    getNumberedCells() {
        const numbered = {};
        let number = 1;

        for (let placedWord of this.placedWords) {
            const key = `${placedWord.row}-${placedWord.col}`;
            if (!numbered[key]) {
                numbered[key] = number++;
                placedWord.number = numbered[key];
            }
        }

        return numbered;
    }

    generateClues() {
        const cluesContainer = document.getElementById('cluesContainer');
        const acrossClues = [];
        const downClues = [];

        for (let placedWord of this.placedWords) {
            const clue = `${placedWord.number}. ${this.generateClueText(placedWord.word)} (${placedWord.word.length})`;
            
            if (placedWord.direction === 'horizontal') {
                acrossClues.push(clue);
            } else {
                downClues.push(clue);
            }
        }

        const cluesHTML = `
            <div class="clues-section">
                <h3>Across</h3>
                <ol>
                    ${acrossClues.map(clue => `<li>${clue}</li>`).join('')}
                </ol>
            </div>
            <div class="clues-section">
                <h3>Down</h3>
                <ol>
                    ${downClues.map(clue => `<li>${clue}</li>`).join('')}
                </ol>
            </div>
        `;

        cluesContainer.innerHTML = cluesHTML;
    }

    generateClueText(word) {
        // Dizionario delle definizioni italiane - molto ampliato
        const clues = {
            // Animali - espanso
            'GATTO': 'Felino domestico che fa le fusa',
            'CANE': 'Il migliore amico dell\'uomo',
            'CAVALLO': 'Animale che galoppa',
            'LEONE': 'Re della savana',
            'PESCE': 'Nuota nel mare',
            'UCCELLO': 'Vola nel cielo',
            'MUCCA': 'Produce il latte',
            'PECORA': 'Ha la lana',
            'CONIGLIO': 'Ha le orecchie lunghe',
            'ELEFANTE': 'Ha la proboscide',
            
            // Famiglia e persone
            'MADRE': 'Genitore femminile',
            'PADRE': 'Genitore maschile',
            'FIGLIO': 'Discendente maschio',
            'FIGLIA': 'Discendente femmina',
            'NONNO': 'Padre del padre',
            'NONNA': 'Madre della madre',
            'ZIO': 'Fratello del genitore',
            'ZIA': 'Sorella del genitore',
            'MARITO': 'Sposo',
            'MOGLIE': 'Sposa',
            'FRATELLO': 'Figlio maschio degli stessi genitori',
            'SORELLA': 'Figlia femmina degli stessi genitori',
            
            // Corpo umano
            'TESTA': 'Parte superiore del corpo',
            'OCCHIO': 'Organo della vista',
            'NASO': 'Organo dell\'olfatto',
            'BOCCA': 'Si usa per mangiare',
            'ORECCHIO': 'Organo dell\'udito',
            'MANO': 'Estremit\u00e0 del braccio',
            'PIEDE': 'Estremit\u00e0 della gamba',
            'BRACCIO': 'Arto superiore',
            'GAMBA': 'Arto inferiore',
            'CUORE': 'Organo che pompa il sangue',
            'CERVELLO': 'Organo del pensiero',
            
            // Casa e oggetti
            'CASA': 'Abitazione',
            'TAVOLO': 'Mobile per mangiare',
            'SEDIA': 'Mobile per sedersi',
            'LETTO': 'Mobile per dormire',
            'PORTA': 'Ingresso della stanza',
            'FINESTRA': 'Apertura nel muro',
            'CUCINA': 'Stanza per cucinare',
            'BAGNO': 'Stanza per lavarsi',
            'CAMERA': 'Stanza da letto',
            'GIARDINO': 'Spazio verde',
            'AUTOMOBILE': 'Mezzo di trasporto',
            'BICICLETTA': 'Mezzo a due ruote',
            'LIBRO': 'Si legge',
            'TELEFONO': 'Serve per chiamare',
            
            // Cibo
            'PANE': 'Alimento base',
            'PASTA': 'Specialit\u00e0 italiana',
            'PIZZA': 'Piatto tipico napoletano',
            'FORMAGGIO': 'Derivato del latte',
            'CARNE': 'Proteina animale',
            'VERDURA': 'Ortaggio',
            'FRUTTA': 'Dolce della natura',
            'ACQUA': 'Liquido vitale',
            'VINO': 'Bevanda dell\'uva',
            'LATTE': 'Bianco nutrimento',
            'UOVO': 'Prodotto della gallina',
            'PESCE': 'Nuota nei mari',
            'RISO': 'Cereale orientale',
            'ZUCCHERO': 'Dolcifica',
            'SALE': 'Condimento bianco',
            
            // Natura
            'SOLE': 'Stella del giorno',
            'LUNA': 'Satellite della notte',
            'MARE': 'Distesa d\'acqua salata',
            'MONTAGNA': 'Elevazione naturale',
            'FIUME': 'Corso d\'acqua',
            'ALBERO': 'Pianta con tronco',
            'FIORE': 'Parte colorata della pianta',
            'ERBA': 'Vegetazione del prato',
            'CIELO': 'Volta azzurra',
            'NUVOLA': 'Vapore in cielo',
            'PIOGGIA': 'Acqua che cade',
            'VENTO': 'Aria in movimento',
            'FUOCO': 'Elemento che brucia',
            'TERRA': 'Il nostro pianeta',
            'PIETRA': 'Minerale duro',
            
            // Colori
            'ROSSO': 'Colore del sangue',
            'BLU': 'Colore del mare',
            'VERDE': 'Colore dell\'erba',
            'GIALLO': 'Colore del sole',
            'NERO': 'Assenza di colore',
            'BIANCO': 'Colore della neve',
            'ROSA': 'Colore delicato',
            'VIOLA': 'Colore dell\'ametista',
            
            // Tempo e date
            'ANNO': 'Dodici mesi',
            'MESE': 'Parte dell\'anno',
            'GIORNO': 'Ventiquattro ore',
            'NOTTE': 'Periodo buio',
            'MATTINA': 'Inizio della giornata',
            'SERA': 'Fine della giornata',
            'OGGI': 'Questo giorno',
            'IERI': 'Giorno passato',
            'DOMANI': 'Giorno futuro',
            'TEMPO': 'Scorre sempre',
            
            // Emozioni e stati
            'AMORE': 'Sentimento profondo',
            'GIOIA': 'Sentimento di felicit\u00e0',
            'PAURA': 'Sentimento di timore',
            'RABBIA': 'Sentimento di collera',
            'PACE': 'Assenza di guerra',
            'GUERRA': 'Conflitto armato',
            'VITA': 'Esistenza',
            'MORTE': 'Fine dell\'esistenza',
            'SALUTE': 'Stato di benessere',
            'MALATTIA': 'Stato di malessere',
            
            // Numeri (scritti)
            'UNO': 'Primo numero',
            'DUE': 'Secondo numero',
            'TRE': 'Terzo numero',
            'QUATTRO': 'Quarto numero',
            'CINQUE': 'Quinto numero',
            'DIECI': 'Decimo numero',
            'CENTO': 'Centesimo numero',
            
            // Verbi comuni (infiniti)
            'ESSERE': 'Verbo di esistenza',
            'AVERE': 'Verbo di possesso',
            'FARE': 'Verbo di azione',
            'DIRE': 'Verbo di parola',
            'ANDARE': 'Verbo di movimento',
            'VENIRE': 'Verbo di arrivo',
            'VEDERE': 'Verbo di vista',
            'SENTIRE': 'Verbo di udito',
            'MANGIARE': 'Verbo del nutrimento',
            'BERE': 'Verbo del dissetarsi',
            'DORMIRE': 'Verbo del riposo',
            'CORRERE': 'Verbo della velocit\u00e0',
            'CAMMINARE': 'Verbo del movimento lento',
            'PARLARE': 'Verbo della comunicazione',
            'CANTARE': 'Verbo della musica',
            'BALLARE': 'Verbo del movimento ritmico',
            'RIDERE': 'Verbo dell\'allegria',
            'PIANGERE': 'Verbo della tristezza',
            'STUDIARE': 'Verbo dell\'apprendimento',
            'LAVORARE': 'Verbo del lavoro',
            
            // Aggettivi
            'GRANDE': 'Di dimensioni ampie',
            'PICCOLO': 'Di dimensioni ridotte',
            'ALTO': 'Di statura elevata',
            'BASSO': 'Di statura ridotta',
            'LUNGO': 'Di lunghezza estesa',
            'CORTO': 'Di lunghezza ridotta',
            'NUOVO': 'Appena fatto',
            'VECCHIO': 'Di et\u00e0 avanzata',
            'GIOVANE': 'Di poca et\u00e0',
            'BELLO': 'Di aspetto gradevole',
            'BRUTTO': 'Di aspetto sgradevole',
            'BUONO': 'Di qualit\u00e0 positiva',
            'CATTIVO': 'Di qualit\u00e0 negativa',
            'CALDO': 'Di temperatura elevata',
            'FREDDO': 'Di temperatura bassa',
            'VELOCE': 'Di andatura rapida',
            'LENTO': 'Di andatura lenta',
            
            // Mestieri
            'MEDICO': 'Cura i malati',
            'MAESTRO': 'Insegna a scuola',
            'OPERAIO': 'Lavora in fabbrica',
            'CONTADINO': 'Lavora nei campi',
            'CUOCO': 'Prepara i cibi',
            'PITTORE': 'Dipinge quadri',
            'MUSICISTA': 'Suona strumenti',
            'SCRITTORE': 'Scrive libri',
            'POETA': 'Scrive versi',
            'GIORNALISTA': 'Scrive notizie'
        };

        return clues[word] || `Parola: ${word}`;
    }

    downloadPDF() {
        // Simple implementation - you might want to use a proper PDF library
        window.print();
    }
}

// Initialize the crossword generator when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new CrosswordGenerator();
});
