class CrosswordGenerator {
    constructor() {
        this.words = [];
        this.grid = [];
        this.placedWords = [];
        this.gridSize = 15;
        this.maxAttempts = 1000;
        
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
        
        // Select words for the crossword (8-12 words)
        const selectedWords = this.selectWords(10);
        console.log('Selected words:', selectedWords);

        if (selectedWords.length === 0) {
            alert('No words available for crossword generation');
            return;
        }

        // Place the first word horizontally in the center
        const firstWord = selectedWords[0];
        const startRow = Math.floor(this.gridSize / 2);
        const startCol = Math.floor((this.gridSize - firstWord.length) / 2);
        
        this.placeWord(firstWord, startRow, startCol, 'horizontal');

        // Try to place remaining words
        for (let i = 1; i < selectedWords.length; i++) {
            const word = selectedWords[i];
            let placed = false;
            let attempts = 0;

            while (!placed && attempts < this.maxAttempts) {
                const placement = this.findValidPlacement(word);
                if (placement) {
                    this.placeWord(word, placement.row, placement.col, placement.direction);
                    placed = true;
                } else {
                    attempts++;
                }
            }

            if (!placed) {
                console.log(`Could not place word: ${word}`);
            }
        }

        this.renderCrossword();
        this.generateClues();
    }

    selectWords(count) {
        // Sort words by length and frequency considerations
        const sortedWords = [...this.words].sort((a, b) => {
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

    findValidPlacement(word) {
        const directions = ['horizontal', 'vertical'];
        const placements = [];

        // Find all possible intersection points with existing words
        for (let placedWord of this.placedWords) {
            for (let i = 0; i < word.length; i++) {
                for (let j = 0; j < placedWord.word.length; j++) {
                    if (word[i] === placedWord.word[j]) {
                        // Try both directions
                        for (let direction of directions) {
                            if (direction === placedWord.direction) continue; // Must be perpendicular
                            
                            let row, col;
                            if (direction === 'horizontal') {
                                row = placedWord.direction === 'vertical' 
                                    ? placedWord.row + j 
                                    : placedWord.row;
                                col = placedWord.direction === 'vertical' 
                                    ? placedWord.col - i 
                                    : placedWord.col + j - i;
                            } else {
                                row = placedWord.direction === 'horizontal' 
                                    ? placedWord.row - i 
                                    : placedWord.row + j - i;
                                col = placedWord.direction === 'horizontal' 
                                    ? placedWord.col + j 
                                    : placedWord.col;
                            }

                            if (this.canPlaceWord(word, row, col, direction)) {
                                placements.push({ row, col, direction });
                            }
                        }
                    }
                }
            }
        }

        // Return a random valid placement
        return placements.length > 0 ? placements[Math.floor(Math.random() * placements.length)] : null;
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

        // Check for conflicts and ensure proper spacing
        for (let i = 0; i < word.length; i++) {
            const currentRow = direction === 'horizontal' ? row : row + i;
            const currentCol = direction === 'horizontal' ? col + i : col;
            const currentCell = this.grid[currentRow][currentCol];

            if (currentCell !== '' && currentCell !== word[i]) {
                return false; // Conflict
            }

            // Check adjacent cells for proper spacing (except at intersections)
            if (currentCell === '') {
                const adjacentCells = this.getAdjacentCells(currentRow, currentCol, direction);
                for (let adjCell of adjacentCells) {
                    if (this.grid[adjCell.row] && this.grid[adjCell.row][adjCell.col] && 
                        this.grid[adjCell.row][adjCell.col] !== '') {
                        return false; // Too close to another word
                    }
                }
            }
        }

        // Check cells before and after the word
        if (direction === 'horizontal') {
            if (col > 0 && this.grid[row][col - 1] !== '') return false;
            if (col + word.length < this.gridSize && this.grid[row][col + word.length] !== '') return false;
        } else {
            if (row > 0 && this.grid[row - 1][col] !== '') return false;
            if (row + word.length < this.gridSize && this.grid[row + word.length][col] !== '') return false;
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

        console.log(`Placed word: ${word} at (${row}, ${col}) ${direction}`);
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
        // Simple clue generation - you can expand this with a proper clue database
        const clues = {
            'CASA': 'Dwelling place',
            'SOLE': 'Star in our solar system',
            'MARE': 'Large body of water',
            'VITA': 'Existence',
            'AMORE': 'Strong affection',
            'TEMPO': 'Duration',
            'GIORNO': 'Day',
            'NOTTE': 'Night'
        };

        return clues[word] || `Word: ${word}`;
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
