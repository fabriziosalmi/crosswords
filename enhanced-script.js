/**
 * Enhanced Crossword Generator
 * Modern UI/UX with improved functionality
 */

class EnhancedCrosswordGenerator {
    constructor() {
        // Core properties
        this.words = [];
        this.grid = [];
        this.placedWords = [];
        this.gridSize = 15;
        this.maxAttempts = 1000;
        this.usedWords = new Set();
        this.theme = 'general';
        this.maxWords = 12;
        
        // UI state
        this.isGenerating = false;
        this.currentFocusedCell = null;
        this.gameStartTime = null;
        this.timerInterval = null;
        this.gameStats = {
            correctCells: 0,
            totalCells: 0,
            hintsUsed: 0,
            errors: 0
        };
        
        this.initializeApp();
    }
    
    async initializeApp() {
        await this.loadWords();
        this.initializeEventListeners();
        this.initializeKeyboardNavigation();
        this.showStatusMessage('Generatore di cruciverba caricato con successo!', 'success');
    }
    
    async loadWords() {
        try {
            // Try to load from the existing word file
            let words = [];
            try {
                const response = await fetch('data/italian_words.txt');
                if (response.ok) {
                    const text = await response.text();
                    words = text.split('\n')
                        .map(line => line.trim().split(':')[0])
                        .filter(word => word && word.length >= 3 && word.length <= 12);
                }
            } catch (e) {
                console.warn('Could not load external word file, using built-in dictionary');
            }
            
            // Fallback to extensive built-in dictionary if needed
            if (words.length === 0) {
                words = this.getBuiltInWords();
            }
            
            this.words = words.map(word => word.trim().toUpperCase())
                .filter(word => word.length >= 3 && word.length <= 12)
                .filter(word => /^[A-Z]+$/.test(word));
            
            console.log(`Loaded ${this.words.length} words`);
        } catch (error) {
            console.error('Error loading words:', error);
            this.words = this.getBuiltInWords();
            this.showStatusMessage('Utilizzando dizionario integrato', 'warning');
        }
    }
    
    getBuiltInWords() {
        // Extensive Italian word dictionary
        return [
            'CASA', 'SOLE', 'MARE', 'VITA', 'AMORE', 'TEMPO', 'GIORNO', 'NOTTE',
            'GATTO', 'CANE', 'CAVALLO', 'LEONE', 'PESCE', 'UCCELLO', 'MUCCA', 'PECORA',
            'MADRE', 'PADRE', 'FIGLIO', 'FIGLIA', 'NONNO', 'NONNA', 'ZIO', 'ZIA',
            'TESTA', 'OCCHIO', 'NASO', 'BOCCA', 'MANO', 'PIEDE', 'BRACCIO', 'GAMBA',
            'TAVOLO', 'SEDIA', 'LETTO', 'PORTA', 'FINESTRA', 'CUCINA', 'BAGNO', 'CAMERA',
            'PANE', 'PASTA', 'PIZZA', 'CARNE', 'VERDURA', 'FRUTTA', 'ACQUA', 'VINO',
            'LUNA', 'STELLA', 'MONTAGNA', 'FIUME', 'ALBERO', 'FIORE', 'ERBA', 'CIELO',
            'ROSSO', 'BLU', 'VERDE', 'GIALLO', 'NERO', 'BIANCO', 'ROSA', 'VIOLA',
            'ANNO', 'MESE', 'OGGI', 'IERI', 'DOMANI', 'MATTINA', 'SERA', 'INVERNO',
            'GRANDE', 'PICCOLO', 'ALTO', 'BASSO', 'LUNGO', 'CORTO', 'NUOVO', 'VECCHIO',
            'ESSERE', 'AVERE', 'FARE', 'DIRE', 'ANDARE', 'VENIRE', 'VEDERE', 'SENTIRE',
            'MEDICO', 'MAESTRO', 'CUOCO', 'PITTORE', 'MUSICISTA', 'SCRITTORE', 'POETA'
        ];
    }
    
    initializeEventListeners() {
        // Generation controls
        document.getElementById('generateBtn').addEventListener('click', () => this.generateCrossword());
        document.getElementById('resetBtn').addEventListener('click', () => this.resetGame());
        
        // Game controls
        document.getElementById('checkBtn').addEventListener('click', () => this.checkSolution());
        document.getElementById('hintBtn').addEventListener('click', () => this.provideHint());
        document.getElementById('clearBtn').addEventListener('click', () => this.clearGrid());
        
        // Export controls
        document.getElementById('printBtn').addEventListener('click', () => this.printCrossword());
        document.getElementById('saveBtn').addEventListener('click', () => this.saveCrossword());
        
        // Settings controls
        document.getElementById('gridSize').addEventListener('change', (e) => {
            this.gridSize = parseInt(e.target.value);
        });
        
        document.getElementById('maxWords').addEventListener('change', (e) => {
            this.maxWords = parseInt(e.target.value);
        });
        
        document.getElementById('theme').addEventListener('change', (e) => {
            this.theme = e.target.value;
        });
        
        // UI controls
        document.getElementById('toggleClues').addEventListener('click', () => this.toggleClues());
        
        // Modal controls
        document.getElementById('modalClose').addEventListener('click', () => this.hideModal());
        document.getElementById('newGameBtn').addEventListener('click', () => {
            this.hideModal();
            this.generateCrossword();
        });
        
        // Click outside modal to close
        document.getElementById('completionModal').addEventListener('click', (e) => {
            if (e.target.id === 'completionModal') {
                this.hideModal();
            }
        });
    }
    
    initializeKeyboardNavigation() {
        document.addEventListener('keydown', (e) => {
            if (this.currentFocusedCell) {
                this.handleKeyboardNavigation(e);
            }
        });
    }
    
    handleKeyboardNavigation(e) {
        if (!this.currentFocusedCell) return;
        
        const cell = this.currentFocusedCell;
        const row = parseInt(cell.dataset.row);
        const col = parseInt(cell.dataset.col);
        let nextCell = null;
        
        switch (e.key) {
            case 'ArrowUp':
                e.preventDefault();
                nextCell = this.findNextCell(row - 1, col);
                break;
            case 'ArrowDown':
                e.preventDefault();
                nextCell = this.findNextCell(row + 1, col);
                break;
            case 'ArrowLeft':
                e.preventDefault();
                nextCell = this.findNextCell(row, col - 1);
                break;
            case 'ArrowRight':
                e.preventDefault();
                nextCell = this.findNextCell(row, col + 1);
                break;
            case 'Backspace':
                if (cell.value === '') {
                    // Move to previous cell
                    const prevCell = this.findPreviousCell(row, col);
                    if (prevCell) {
                        prevCell.focus();
                        prevCell.value = '';
                    }
                }
                break;
            case 'Delete':
                cell.value = '';
                this.updateProgress();
                break;
        }
        
        if (nextCell) {
            nextCell.focus();
        }
    }
    
    findNextCell(row, col) {
        if (row >= 0 && row < this.gridSize && col >= 0 && col < this.gridSize) {
            const cell = document.querySelector(`input[data-row="${row}"][data-col="${col}"]`);
            return cell;
        }
        return null;
    }
    
    findPreviousCell(row, col) {
        // Look for the previous cell in reading order
        for (let r = row; r >= 0; r--) {
            for (let c = (r === row ? col - 1 : this.gridSize - 1); c >= 0; c--) {
                const cell = this.findNextCell(r, c);
                if (cell) return cell;
            }
        }
        return null;
    }
    
    async generateCrossword() {
        if (this.isGenerating) return;
        
        this.isGenerating = true;
        this.updateGenerateButton(true);
        
        try {
            this.showStatusMessage('Generazione cruciverba in corso...', 'info');
            
            // Reset game state
            this.resetGameState();
            
            // Initialize grid
            this.initializeGrid();
            this.usedWords.clear();
            
            // Select words based on theme and settings
            const selectedWords = this.selectWords(this.maxWords * 2);
            
            if (selectedWords.length === 0) {
                throw new Error('Nessuna parola disponibile per la generazione');
            }
            
            // Generate crossword with improved algorithm
            await this.generateCrosswordGrid(selectedWords);
            
            // Render the crossword
            this.renderCrossword();
            this.generateClues();
            
            // Start game timer
            this.startTimer();
            
            this.showStatusMessage(`Cruciverba generato con ${this.placedWords.length} parole!`, 'success');
            
        } catch (error) {
            console.error('Error generating crossword:', error);
            this.showStatusMessage(error.message || 'Errore durante la generazione', 'error');
        } finally {
            this.isGenerating = false;
            this.updateGenerateButton(false);
        }
    }
    
    resetGameState() {
        this.placedWords = [];
        this.gameStats = {
            correctCells: 0,
            totalCells: 0,
            hintsUsed: 0,
            errors: 0
        };
        this.stopTimer();
        this.updateProgress();
    }
    
    initializeGrid() {
        this.grid = Array(this.gridSize).fill(null).map(() => 
            Array(this.gridSize).fill('')
        );
    }
    
    selectWords(count) {
        let availableWords = [...this.words];
        
        // Filter by theme if not general
        if (this.theme !== 'general') {
            availableWords = this.filterWordsByTheme(availableWords);
        }
        
        // Sort by preference (length, common letters, etc.)
        availableWords = availableWords.sort((a, b) => {
            return this.getWordScore(b) - this.getWordScore(a);
        });
        
        return availableWords.slice(0, count);
    }
    
    filterWordsByTheme(words) {
        const themeWords = {
            animals: ['GATTO', 'CANE', 'CAVALLO', 'LEONE', 'PESCE', 'UCCELLO', 'MUCCA', 'PECORA', 'CONIGLIO', 'ELEFANTE'],
            family: ['MADRE', 'PADRE', 'FIGLIO', 'FIGLIA', 'NONNO', 'NONNA', 'ZIO', 'ZIA', 'MARITO', 'MOGLIE', 'FRATELLO', 'SORELLA'],
            food: ['PANE', 'PASTA', 'PIZZA', 'FORMAGGIO', 'CARNE', 'VERDURA', 'FRUTTA', 'ACQUA', 'VINO', 'LATTE', 'UOVO', 'PESCE', 'RISO'],
            nature: ['SOLE', 'LUNA', 'MARE', 'MONTAGNA', 'FIUME', 'ALBERO', 'FIORE', 'ERBA', 'CIELO', 'NUVOLA', 'PIOGGIA', 'VENTO']
        };
        
        const themeSet = new Set(themeWords[this.theme] || []);
        return words.filter(word => themeSet.has(word) || themeWords[this.theme] === undefined);
    }
    
    getWordScore(word) {
        let score = 0;
        
        // Prefer medium length words (4-8 chars)
        if (word.length >= 4 && word.length <= 8) {
            score += 10;
        } else if (word.length >= 3 && word.length <= 10) {
            score += 5;
        }
        
        // Bonus for common vowels and consonants
        const commonLetters = 'AEIOURNSTL';
        for (let char of word) {
            if (commonLetters.includes(char)) {
                score += 1;
            }
        }
        
        return score;
    }
    
    async generateCrosswordGrid(selectedWords) {
        // Place the first word horizontally in the center
        const firstWord = selectedWords[0];
        const startRow = Math.floor(this.gridSize / 2);
        const startCol = Math.floor((this.gridSize - firstWord.length) / 2);
        
        if (!this.placeWord(firstWord, startRow, startCol, 'horizontal')) {
            throw new Error('Impossibile posizionare la prima parola');
        }
        
        // Try to place remaining words
        let wordsPlaced = 1;
        let totalAttempts = 0;
        const maxTotalAttempts = selectedWords.length * 50;
        
        while (wordsPlaced < this.maxWords && totalAttempts < maxTotalAttempts) {
            const availableWords = selectedWords.filter(word => !this.usedWords.has(word));
            
            if (availableWords.length === 0) break;
            
            let wordPlaced = false;
            
            for (const word of availableWords) {
                const placements = this.findAllValidPlacements(word);
                
                if (placements.length > 0) {
                    const placement = placements[Math.floor(Math.random() * placements.length)];
                    
                    if (this.placeWord(word, placement.row, placement.col, placement.direction)) {
                        wordsPlaced++;
                        wordPlaced = true;
                        break;
                    }
                }
            }
            
            if (!wordPlaced) {
                totalAttempts++;
            }
            
            // Add small delay for UI responsiveness
            if (totalAttempts % 10 === 0) {
                await this.delay(1);
            }
        }
        
        if (this.placedWords.length < 3) {
            throw new Error('Impossibile generare un cruciverba valido con le impostazioni correnti');
        }
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    findAllValidPlacements(word) {
        const placements = [];
        
        for (let placedWord of this.placedWords) {
            for (let i = 0; i < word.length; i++) {
                for (let j = 0; j < placedWord.word.length; j++) {
                    if (word[i] === placedWord.word[j]) {
                        if (placedWord.direction === 'horizontal') {
                            const newRow = placedWord.row - i;
                            const newCol = placedWord.col + j;
                            
                            if (this.canPlaceWord(word, newRow, newCol, 'vertical')) {
                                placements.push({ row: newRow, col: newCol, direction: 'vertical' });
                            }
                        } else {
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
        
        // Check for conflicts and proper word separation
        return this.checkWordPlacement(word, row, col, direction);
    }
    
    checkWordPlacement(word, row, col, direction) {
        // Check each cell of the word
        for (let i = 0; i < word.length; i++) {
            const currentRow = direction === 'horizontal' ? row : row + i;
            const currentCol = direction === 'horizontal' ? col + i : col;
            const currentCell = this.grid[currentRow][currentCol];
            
            if (currentCell !== '' && currentCell !== word[i]) {
                return false;
            }
        }
        
        // Check word boundaries
        if (direction === 'horizontal') {
            // Check before and after word
            if ((col > 0 && this.grid[row][col - 1] !== '') || 
                (col + word.length < this.gridSize && this.grid[row][col + word.length] !== '')) {
                return false;
            }
            
            // Check parallel cells
            for (let i = 0; i < word.length; i++) {
                const checkCol = col + i;
                if (this.grid[row][checkCol] === '') {
                    if ((row > 0 && this.grid[row - 1][checkCol] !== '') ||
                        (row < this.gridSize - 1 && this.grid[row + 1][checkCol] !== '')) {
                        return false;
                    }
                }
            }
        } else {
            // Vertical placement checks
            if ((row > 0 && this.grid[row - 1][col] !== '') ||
                (row + word.length < this.gridSize && this.grid[row + word.length][col] !== '')) {
                return false;
            }
            
            for (let i = 0; i < word.length; i++) {
                const checkRow = row + i;
                if (this.grid[checkRow][col] === '') {
                    if ((col > 0 && this.grid[checkRow][col - 1] !== '') ||
                        (col < this.gridSize - 1 && this.grid[checkRow][col + 1] !== '')) {
                        return false;
                    }
                }
            }
        }
        
        return true;
    }
    
    placeWord(word, row, col, direction) {
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
        
        const numberedCells = this.getNumberedCells();
        this.gameStats.totalCells = 0;
        
        for (let i = 0; i < this.gridSize; i++) {
            const row = document.createElement('tr');
            
            for (let j = 0; j < this.gridSize; j++) {
                const cell = document.createElement('td');
                
                if (this.grid[i][j] !== '') {
                    cell.className = 'crossword-cell';
                    this.gameStats.totalCells++;
                    
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
                    input.dataset.row = i;
                    input.dataset.col = j;
                    input.setAttribute('aria-label', `Cella ${i + 1}, ${j + 1}`);
                    
                    // Add event listeners
                    input.addEventListener('input', (e) => this.handleCellInput(e));
                    input.addEventListener('focus', (e) => this.handleCellFocus(e));
                    input.addEventListener('blur', (e) => this.handleCellBlur(e));
                    
                    cell.appendChild(input);
                } else {
                    cell.className = 'empty-cell';
                    cell.setAttribute('aria-hidden', 'true');
                }
                
                row.appendChild(cell);
            }
            
            table.appendChild(row);
        }
        
        container.innerHTML = '';
        container.appendChild(table);
        this.updateProgress();
    }
    
    handleCellInput(e) {
        const input = e.target;
        const value = input.value.toUpperCase();
        
        if (value && !/^[A-Z]$/.test(value)) {
            input.value = '';
            return;
        }
        
        input.value = value;
        
        // Move to next cell if filled
        if (value) {
            this.moveToNextCell(input);
        }
        
        this.updateProgress();
        this.checkCellCorrectness(input);
    }
    
    handleCellFocus(e) {
        this.currentFocusedCell = e.target;
        e.target.closest('td').classList.add('focused');
        this.highlightRelatedCells(e.target);
    }
    
    handleCellBlur(e) {
        e.target.closest('td').classList.remove('focused');
        this.clearCellHighlights();
    }
    
    highlightRelatedCells(input) {
        const row = parseInt(input.dataset.row);
        const col = parseInt(input.dataset.col);
        
        // Find and highlight the word this cell belongs to
        for (let placedWord of this.placedWords) {
            if (this.cellBelongsToWord(row, col, placedWord)) {
                this.highlightWord(placedWord);
                break;
            }
        }
    }
    
    cellBelongsToWord(row, col, placedWord) {
        if (placedWord.direction === 'horizontal') {
            return row === placedWord.row && 
                   col >= placedWord.col && 
                   col < placedWord.col + placedWord.word.length;
        } else {
            return col === placedWord.col && 
                   row >= placedWord.row && 
                   row < placedWord.row + placedWord.word.length;
        }
    }
    
    highlightWord(placedWord) {
        for (let i = 0; i < placedWord.word.length; i++) {
            const cellRow = placedWord.direction === 'horizontal' ? placedWord.row : placedWord.row + i;
            const cellCol = placedWord.direction === 'horizontal' ? placedWord.col + i : placedWord.col;
            const cell = document.querySelector(`input[data-row="${cellRow}"][data-col="${cellCol}"]`);
            if (cell) {
                cell.closest('td').classList.add('word-highlighted');
            }
        }
    }
    
    clearCellHighlights() {
        document.querySelectorAll('.word-highlighted').forEach(cell => {
            cell.classList.remove('word-highlighted');
        });
    }
    
    moveToNextCell(currentInput) {
        const row = parseInt(currentInput.dataset.row);
        const col = parseInt(currentInput.dataset.col);
        
        // Try to move to the next cell in the word
        let nextCell = this.findNextCell(row, col + 1);
        if (!nextCell) {
            nextCell = this.findNextCell(row + 1, col);
        }
        if (!nextCell) {
            nextCell = this.findNextEmptyCell();
        }
        
        if (nextCell) {
            nextCell.focus();
        }
    }
    
    findNextEmptyCell() {
        const inputs = document.querySelectorAll('.crossword-cell input');
        for (let input of inputs) {
            if (!input.value) {
                return input;
            }
        }
        return null;
    }
    
    checkCellCorrectness(input) {
        const isCorrect = input.value === input.dataset.answer;
        const cell = input.closest('td');
        
        cell.classList.remove('correct', 'incorrect');
        
        if (input.value) {
            if (isCorrect) {
                cell.classList.add('correct');
                this.gameStats.correctCells++;
            } else {
                cell.classList.add('incorrect');
                this.gameStats.errors++;
                
                // Remove incorrect class after animation
                setTimeout(() => {
                    cell.classList.remove('incorrect');
                }, 500);
            }
        }
    }
    
    getNumberedCells() {
        const numbered = {};
        let number = 1;
        
        for (let placedWord of this.placedWords) {
            const key = `${placedWord.row}-${placedWord.col}`;
            if (!numbered[key]) {
                numbered[key] = number++;
                placedWord.number = numbered[key];
            } else {
                placedWord.number = numbered[key];
            }
        }
        
        return numbered;
    }
    
    generateClues() {
        const container = document.getElementById('cluesContainer');
        const acrossClues = [];
        const downClues = [];
        
        for (let placedWord of this.placedWords) {
            const clue = {
                number: placedWord.number,
                text: this.generateClueText(placedWord.word),
                length: placedWord.word.length,
                word: placedWord.word
            };
            
            if (placedWord.direction === 'horizontal') {
                acrossClues.push(clue);
            } else {
                downClues.push(clue);
            }
        }
        
        // Sort by number
        acrossClues.sort((a, b) => a.number - b.number);
        downClues.sort((a, b) => a.number - b.number);
        
        const cluesHTML = `
            <div class="clues-group">
                <h3>Orizzontali</h3>
                <ul class="clues-list">
                    ${acrossClues.map(clue => `
                        <li class="clue-item" data-word="${clue.word}" data-number="${clue.number}">
                            <span class="clue-number">${clue.number}.</span>
                            <span class="clue-text">${clue.text}</span>
                            <span class="clue-length">(${clue.length})</span>
                        </li>
                    `).join('')}
                </ul>
            </div>
            <div class="clues-group">
                <h3>Verticali</h3>
                <ul class="clues-list">
                    ${downClues.map(clue => `
                        <li class="clue-item" data-word="${clue.word}" data-number="${clue.number}">
                            <span class="clue-number">${clue.number}.</span>
                            <span class="clue-text">${clue.text}</span>
                            <span class="clue-length">(${clue.length})</span>
                        </li>
                    `).join('')}
                </ul>
            </div>
        `;
        
        container.innerHTML = cluesHTML;
        
        // Add click handlers to clues
        container.querySelectorAll('.clue-item').forEach(item => {
            item.addEventListener('click', () => this.focusOnWord(item.dataset.word, item.dataset.number));
        });
    }
    
    focusOnWord(word, number) {
        const placedWord = this.placedWords.find(pw => pw.word === word && pw.number == number);
        if (placedWord) {
            const firstCell = document.querySelector(`input[data-row="${placedWord.row}"][data-col="${placedWord.col}"]`);
            if (firstCell) {
                firstCell.focus();
            }
        }
    }
    
    generateClueText(word) {
        const clues = {
            // Animali
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
            
            // Famiglia
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
            'MANO': 'Estremità del braccio',
            'PIEDE': 'Estremità della gamba',
            'BRACCIO': 'Arto superiore',
            'GAMBA': 'Arto inferiore',
            'CUORE': 'Organo che pompa il sangue',
            
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
            
            // Cibo
            'PANE': 'Alimento base',
            'PASTA': 'Specialità italiana',
            'PIZZA': 'Piatto tipico napoletano',
            'FORMAGGIO': 'Derivato del latte',
            'CARNE': 'Proteina animale',
            'VERDURA': 'Ortaggio',
            'FRUTTA': 'Dolce della natura',
            'ACQUA': 'Liquido vitale',
            'VINO': 'Bevanda dell\'uva',
            'LATTE': 'Bianco nutrimento',
            'UOVO': 'Prodotto della gallina',
            'RISO': 'Cereale orientale',
            
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
            'TERRA': 'Il nostro pianeta',
            
            // Colori
            'ROSSO': 'Colore del sangue',
            'BLU': 'Colore del mare',
            'VERDE': 'Colore dell\'erba',
            'GIALLO': 'Colore del sole',
            'NERO': 'Assenza di colore',
            'BIANCO': 'Colore della neve',
            'ROSA': 'Colore delicato',
            'VIOLA': 'Colore dell\'ametista',
            
            // Tempo
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
            
            // Aggettivi
            'GRANDE': 'Di dimensioni ampie',
            'PICCOLO': 'Di dimensioni ridotte',
            'ALTO': 'Di statura elevata',
            'BASSO': 'Di statura ridotta',
            'LUNGO': 'Di lunghezza estesa',
            'CORTO': 'Di lunghezza ridotta',
            'NUOVO': 'Appena fatto',
            'VECCHIO': 'Di età avanzata',
            'GIOVANE': 'Di poca età',
            'BELLO': 'Di aspetto gradevole',
            'BRUTTO': 'Di aspetto sgradevole',
            'BUONO': 'Di qualità positiva',
            'CATTIVO': 'Di qualità negativa',
            'CALDO': 'Di temperatura elevata',
            'FREDDO': 'Di temperatura bassa',
            
            // Verbi
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
            'CORRERE': 'Verbo della velocità',
            'CAMMINARE': 'Verbo del movimento lento',
            'PARLARE': 'Verbo della comunicazione',
            
            // Mestieri
            'MEDICO': 'Cura i malati',
            'MAESTRO': 'Insegna a scuola',
            'CUOCO': 'Prepara i cibi',
            'PITTORE': 'Dipinge quadri',
            'MUSICISTA': 'Suona strumenti',
            'SCRITTORE': 'Scrive libri',
            'POETA': 'Scrive versi'
        };
        
        return clues[word] || `Definizione di ${word}`;
    }
    
    // UI Control Methods
    updateGenerateButton(isLoading) {
        const btn = document.getElementById('generateBtn');
        const icon = btn.querySelector('i');
        const text = btn.querySelector('span') || btn;
        
        if (isLoading) {
            btn.disabled = true;
            icon.className = 'fas fa-spinner fa-spin';
            if (text !== btn) text.textContent = 'Generazione...';
        } else {
            btn.disabled = false;
            icon.className = 'fas fa-magic';
            if (text !== btn) text.textContent = 'Genera Cruciverba';
        }
    }
    
    startTimer() {
        this.gameStartTime = Date.now();
        this.timerInterval = setInterval(() => {
            this.updateTimer();
        }, 1000);
    }
    
    stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }
    
    updateTimer() {
        if (!this.gameStartTime) return;
        
        const elapsed = Math.floor((Date.now() - this.gameStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        
        document.getElementById('timeDisplay').textContent = 
            `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }
    
    updateProgress() {
        const inputs = document.querySelectorAll('.crossword-cell input');
        const filledCells = Array.from(inputs).filter(input => input.value).length;
        const totalCells = inputs.length;
        
        const percentage = totalCells > 0 ? Math.round((filledCells / totalCells) * 100) : 0;
        
        document.getElementById('progressFill').style.width = percentage + '%';
        document.getElementById('progressText').textContent = percentage + '%';
        
        // Check if game is completed
        if (percentage === 100) {
            this.checkGameCompletion();
        }
    }
    
    checkGameCompletion() {
        const inputs = document.querySelectorAll('.crossword-cell input');
        const allCorrect = Array.from(inputs).every(input => 
            input.value === input.dataset.answer
        );
        
        if (allCorrect) {
            this.stopTimer();
            this.showCompletionModal();
        }
    }
    
    checkSolution() {
        const inputs = document.querySelectorAll('.crossword-cell input');
        let correct = 0;
        let total = 0;
        
        inputs.forEach(input => {
            total++;
            const cell = input.closest('td');
            cell.classList.remove('correct', 'incorrect');
            
            if (input.value === input.dataset.answer) {
                correct++;
                cell.classList.add('correct');
            } else if (input.value) {
                cell.classList.add('incorrect');
                setTimeout(() => cell.classList.remove('incorrect'), 1000);
            }
        });
        
        const percentage = Math.round((correct / total) * 100);
        this.showStatusMessage(`Soluzione verificata: ${percentage}% corretto (${correct}/${total})`, 
                               percentage === 100 ? 'success' : 'info');
        
        if (percentage === 100) {
            this.stopTimer();
            this.showCompletionModal();
        }
    }
    
    provideHint() {
        const emptyInputs = Array.from(document.querySelectorAll('.crossword-cell input'))
            .filter(input => !input.value);
        
        if (emptyInputs.length === 0) {
            this.showStatusMessage('Tutte le celle sono già compilate!', 'info');
            return;
        }
        
        const randomInput = emptyInputs[Math.floor(Math.random() * emptyInputs.length)];
        randomInput.value = randomInput.dataset.answer;
        randomInput.closest('td').classList.add('correct');
        
        this.gameStats.hintsUsed++;
        this.updateProgress();
        this.showStatusMessage('Suggerimento fornito!', 'success');
    }
    
    clearGrid() {
        const inputs = document.querySelectorAll('.crossword-cell input');
        inputs.forEach(input => {
            input.value = '';
            input.closest('td').classList.remove('correct', 'incorrect');
        });
        
        this.updateProgress();
        this.showStatusMessage('Griglia pulita', 'info');
    }
    
    resetGame() {
        this.stopTimer();
        this.clearGrid();
        this.resetGameState();
        
        document.getElementById('crosswordContainer').innerHTML = `
            <div class="empty-state">
                <i class="fas fa-puzzle-piece"></i>
                <p>Premi "Genera Cruciverba" per iniziare</p>
            </div>
        `;
        
        document.getElementById('cluesContainer').innerHTML = `
            <div class="empty-state">
                <i class="fas fa-list-alt"></i>
                <p>Le definizioni appariranno qui</p>
            </div>
        `;
        
        document.getElementById('timeDisplay').textContent = '00:00';
        this.showStatusMessage('Gioco resettato', 'info');
    }
    
    toggleClues() {
        const container = document.getElementById('cluesContainer');
        const button = document.getElementById('toggleClues');
        const icon = button.querySelector('i');
        
        if (container.style.display === 'none') {
            container.style.display = '';
            icon.className = 'fas fa-eye';
            button.title = 'Nascondi definizioni';
        } else {
            container.style.display = 'none';
            icon.className = 'fas fa-eye-slash';
            button.title = 'Mostra definizioni';
        }
    }
    
    printCrossword() {
        // Hide controls before printing
        const controlPanel = document.getElementById('controlPanel');
        const originalDisplay = controlPanel.style.display;
        
        controlPanel.style.display = 'none';
        
        window.print();
        
        // Restore controls after printing
        setTimeout(() => {
            controlPanel.style.display = originalDisplay;
        }, 1000);
    }
    
    saveCrossword() {
        const crosswordHTML = this.generateCrosswordHTML();
        const blob = new Blob([crosswordHTML], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `cruciverba_${new Date().toISOString().split('T')[0]}.html`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.showStatusMessage('Cruciverba salvato come HTML', 'success');
    }
    
    generateCrosswordHTML() {
        const crosswordContent = document.getElementById('crosswordContainer').outerHTML;
        const cluesContent = document.getElementById('cluesContainer').outerHTML;
        
        return `<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cruciverba - ${new Date().toLocaleDateString()}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .crossword-grid { border-collapse: separate; border-spacing: 2px; margin: 0 auto 30px; background: #ccc; padding: 4px; }
        .crossword-grid td { width: 35px; height: 35px; position: relative; border-radius: 4px; }
        .crossword-cell { background: white; border: 2px solid #ddd; }
        .empty-cell { background: #333; }
        .crossword-cell input { width: 100%; height: 100%; border: none; text-align: center; font-size: 18px; font-weight: 600; text-transform: uppercase; }
        .cell-number { position: absolute; top: 2px; left: 2px; font-size: 10px; font-weight: 600; color: #667eea; background: white; padding: 1px 2px; border-radius: 2px; }
        .clues-group { margin-bottom: 30px; }
        .clues-group h3 { color: #333; border-bottom: 2px solid #667eea; padding-bottom: 5px; }
        .clues-list { list-style: none; padding: 0; }
        .clue-item { margin-bottom: 12px; padding: 8px 16px; background: #f7fafc; border-left: 4px solid #e6f3ff; }
        .clue-number { font-weight: 600; color: #667eea; margin-right: 8px; }
        .clue-length { color: #718096; font-size: 0.875rem; margin-left: 8px; }
        @media print {
            .crossword-cell input { color: transparent !important; }
        }
    </style>
</head>
<body>
    <h1 style="text-align: center;">Cruciverba</h1>
    ${crosswordContent}
    ${cluesContent}
</body>
</html>`;
    }
    
    showCompletionModal() {
        const modal = document.getElementById('completionModal');
        const elapsed = Math.floor((Date.now() - this.gameStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        
        document.getElementById('finalTime').textContent = 
            `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        
        const accuracy = Math.round(((this.gameStats.totalCells - this.gameStats.errors) / this.gameStats.totalCells) * 100);
        document.getElementById('finalAccuracy').textContent = accuracy + '%';
        
        modal.classList.add('show');
    }
    
    hideModal() {
        document.getElementById('completionModal').classList.remove('show');
    }
    
    showStatusMessage(message, type = 'info') {
        const container = document.getElementById('statusMessages');
        const messageEl = document.createElement('div');
        messageEl.className = `status-message ${type}`;
        messageEl.innerHTML = `
            ${message}
            <button class="close-btn">&times;</button>
        `;
        
        const closeBtn = messageEl.querySelector('.close-btn');
        closeBtn.addEventListener('click', () => {
            messageEl.remove();
        });
        
        container.appendChild(messageEl);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (messageEl.parentNode) {
                messageEl.remove();
            }
        }, 5000);
    }
}

// Initialize the enhanced crossword generator when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new EnhancedCrosswordGenerator();
});
