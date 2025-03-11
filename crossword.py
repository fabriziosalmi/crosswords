import argparse
import logging
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from typing import List, Tuple, Dict, Optional
from threading import Lock
from functools import lru_cache
from collections import defaultdict
import nltk  # For word frequency
from rich.progress import Progress, TaskID
from rich.console import Console
from rich.table import Table
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# --- Constants (Moved to top for clarity and easy modification) ---
DEFAULT_GRID_WIDTH = 15
DEFAULT_GRID_HEIGHT = 15
DEFAULT_BLACK_SQUARE_RATIO = 0.17  # Adjusted for a more reasonable default
DEFAULT_LM_STUDIO_URL = "http://localhost:1234/v1"
DEFAULT_WORDS_FILE = "data/parole.txt"
DEFAULT_OUTPUT_FILENAME = "docs/cruciverba.html"
DEFAULT_MAX_ATTEMPTS = 100  # Per word placement
DEFAULT_TIMEOUT = 180  # Overall timeout
DEFAULT_LLM_TIMEOUT = 30
DEFAULT_LLM_MAX_TOKENS = 64 #increased for better definition quality
DEFAULT_LANGUAGE = "Italian"
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"  # Changed to a reliable open model
MAX_RECURSION_DEPTH = 1000  # Increased, but still a safety net
DEFAULT_BEAM_WIDTH = 10   # Slightly increased
DEFAULT_MAX_BACKTRACK = 300 # Increased
MIN_WORD_LENGTH = 2      # Increased minimum word length
FORBIDDEN_PATTERNS = [
    r'\b{}\b',
    r'{}'.format,
    r'(?i){}'.format,
]
MAX_DEFINITION_ATTEMPTS = 3  # Retry definition generation
DEFINITION_RETRY_DELAY = 2  # Seconds between definition retries

# First, add new constants for difficulty settings
DEFAULT_DIFFICULTY = "medium"  # Options: easy, medium, hard
WORD_FREQUENCY_WEIGHTS = {
    "easy": 0.8,     # Prefer common words
    "medium": 0.5,   # Balanced
    "hard": 0.2      # Prefer rare words
}
MIN_WORD_COUNTS = {
    "easy": 30,
    "medium": 40,
    "hard": 50
}

# Add this constant near the top with other constants
MAX_THREAD_POOL_SIZE = 16  # Limit maximum number of threads

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Compiled Regex Patterns ---
WORD_CLEAN_RE = re.compile(r"[^A-Z]")
DEFINITION_CLEAN_RE = re.compile(r"^\d+\.\s*")
NON_ALPHANUMERIC_RE = re.compile(r"^[^\w]+|[^\w]+$")

# --- Global Variables ---
# Use a thread-safe cache
cache_lock = Lock()
placement_cache: Dict[str, bool] = {}
definition_cache: Dict[str, str] = {}

# --- Utility Functions ---

def print_grid(grid: List[List[str]], placed_words:List[Tuple[str, int, int, str]]=None, console:Optional[Console]=None) -> None:
    """Prints a visual representation of the grid with optional highlighting of placed words."""
    if console is None:
        console = Console()

    table = Table(show_header=False, show_edge=False, padding=0)

    if placed_words is not None:
         placed_coords = set()
         for word, row, col, direction in placed_words:
             for i in range(len(word)):
                 if direction == "across":
                     placed_coords.add((row, col + i))
                 else:
                     placed_coords.add((row + i, col))

    for r_idx,row in enumerate(grid):
        row_display = []
        for c_idx,cell in enumerate(row):
            if cell == "#":
                row_display.append("[white on black]  [/]")
            elif placed_words is not None and (r_idx,c_idx) in placed_coords:
                row_display.append(f"[black on green]{cell.center(2)}[/]")  # Highlight placed letters
            else:
                row_display.append(f"[black on white]{cell.center(2)}[/]")

        table.add_row(*row_display)

    console.print(table)



def calculate_word_frequency(word: str, word_frequencies: Dict[str, float]) -> float:
    """Calculates a word's score based on its frequency (lower score is more common)."""
    return word_frequencies.get(word.lower(), 1e-6)  # Default to a very low frequency if not found


def load_words(filepath: str) -> Tuple[Dict[int, List[str]], Dict[str, float]]:
    """Loads, preprocesses words, groups by length, and calculates frequencies."""
    words_by_length: Dict[int, List[str]] = defaultdict(list)
    word_counts: Dict[str, int] = defaultdict(int)
    total_count = 0

    try:
        with open(filepath, "r", encoding="utf-8") as file:
            for line in file:
                word = line.strip().upper()
                word = WORD_CLEAN_RE.sub("", word)
                if len(word) >= MIN_WORD_LENGTH:
                    words_by_length[len(word)].append(word)
                    word_counts[word.lower()] += 1
                    total_count += 1

        # Calculate frequencies
        word_frequencies: Dict[str, float] = {
            word: count / total_count for word, count in word_counts.items()
        }
        return words_by_length, word_frequencies

    except FileNotFoundError:
        logging.error(f"Word file not found at {filepath}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading or processing words: {e}")
        sys.exit(1)


# --- LangChain and LLM Setup ---

def setup_langchain_llm(lm_studio_url: str, llm_timeout: int, llm_max_tokens: int, model: str) -> ChatOpenAI:
    """Sets up the LangChain LLM with retries, timeout, and error handling."""
    try:
        llm = ChatOpenAI(
            base_url=lm_studio_url,
            api_key="NA",
            model=model,
            temperature=0.7,
            max_tokens=llm_max_tokens,
            timeout=llm_timeout,
        )
        return llm
    except Exception as e:
        logging.error(f"Failed to initialize ChatOpenAI: {e}")
        sys.exit(1)

@lru_cache(maxsize=512) #Memoize definition generation
def generate_definition_langchain(llm: ChatOpenAI, word: str, language: str) -> str:
    """Generates a crossword clue using LangChain, with retries and improved filtering."""
    if word in definition_cache:
        return definition_cache[word]

    prompt_template = """Generate a short, concise crossword clue for: "{word}". Reply in {language}.
    Strict Rules:
    1. Absolutely NO part of the target word in the clue.
    2. Clue must be 10 words or less.
    3. Avoid obvious synonyms.
    4. No direct definitions.
    5. Focus on wordplay, double meanings, or indirect hints.
    Output: Only the clue text, nothing else."""

    prompt = PromptTemplate.from_template(prompt_template)
    output_parser = StrOutputParser()
    chain = (
        {"word": RunnablePassthrough(), "language": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )

    for attempt in range(MAX_DEFINITION_ATTEMPTS):
        try:
            definition = chain.invoke({"word": word, "language": language})
            definition = definition.strip()

            # Enhanced cleaning and filtering
            definition = re.sub(r'(?i)definizione[:\s]*', '', definition)
            definition = re.sub(r'(?i)clue[:\s]*', '', definition)
            definition = re.sub(r'^\d+[\.\)]\s*', '', definition).strip()

            for pattern in FORBIDDEN_PATTERNS:
                if re.search(pattern(word), definition, re.IGNORECASE):
                    raise ValueError("Forbidden word/pattern used in definition.")

            # Check for word parts (3+ letters) - More efficient
            word_lower = word.lower()
            definition_lower = definition.lower()
            if any(word_lower[i:j] in definition_lower for i in range(len(word) - 2) for j in range(i + 3, len(word) + 1)):
                  raise ValueError("Part of the word used in the definition")

            definition_cache[word] = definition
            return definition

        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed for word '{word}': {e}")
            if attempt < MAX_DEFINITION_ATTEMPTS - 1:
                time.sleep(DEFINITION_RETRY_DELAY)

    logging.error(f"Failed to generate definition for '{word}' after {MAX_DEFINITION_ATTEMPTS} attempts.")
    return "Definizione non disponibile"  # Or a more specific fallback

# --- Grid Generation and Manipulation ---

def generate_grid_from_string(grid_string: str) -> Optional[List[List[str]]]:
    """Generates grid from string, with validation."""
    lines = grid_string.strip().split("\n")
    grid: List[List[str]] = []
    for line in lines:
        row = [char for char in line.strip() if char in (".", "#")]
        if len(row) != len(lines[0]):
            logging.error("Inconsistent row length in manual grid.")
            return None
        grid.append(row)

    if not grid or any(len(row) != len(grid[0]) for row in grid):  # Check for rectangular grid
        logging.error("Invalid manual grid: empty or non-rectangular.")
        return None
    return grid


def generate_grid_from_file(filepath: str) -> Optional[List[List[str]]]:
    """Loads grid from file, with error handling."""
    try:
        with open(filepath, "r") as f:
            return generate_grid_from_string(f.read())
    except FileNotFoundError:
        logging.error(f"Grid file not found: {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error reading grid file: {e}")
        return None

def is_valid_grid(grid: List[List[str]]) -> bool:
    """Checks if the grid is valid (rectangular, contains only '.' and '#')."""
    if not grid:
        return False
    width = len(grid[0])
    return all(len(row) == width and all(c in ('.', '#') for c in row) for row in grid)

def generate_grid_random(width: int, height: int, black_square_ratio: float) -> List[List[str]]:
    """Generates a random, symmetrical crossword grid with improved black square placement."""
    grid = [["." for _ in range(width)] for _ in range(height)]
    num_black_squares = int(width * height * black_square_ratio)

    # Helper function for symmetrical placement
    def place_symmetrically(row: int, col: int):
        grid[row][col] = "#"
        grid[height - 1 - row][width - 1 - col] = "#"

    # 1. Central square (for odd dimensions)
    if width % 2 == 1 and height % 2 == 1:
        place_symmetrically(height // 2, width // 2)
        num_black_squares -= 1  # Adjust count if central square is placed

    placed_count = 0
    attempts = 0
    max_attempts = width * height * 5  # Increased attempts, but still limited

    while placed_count < num_black_squares and attempts < max_attempts:
        attempts += 1
        row, col = random.randint(0, height - 1), random.randint(0, width - 1)

        if grid[row][col] == ".":  # Only try to place on empty squares
            # Check for 2x2 blocks *before* placement (more efficient)
            if (row > 0 and col > 0 and grid[row-1][col] == "#" and grid[row][col-1] == "#" and grid[row-1][col-1] == "#") or \
               (row > 0 and col < width - 1 and grid[row-1][col] == "#" and grid[row][col+1] == "#" and grid[row-1][col+1] == "#") or \
               (row < height - 1 and col > 0 and grid[row+1][col] == "#" and grid[row][col-1] == "#" and grid[row+1][col-1] == "#") or \
               (row < height - 1 and col < width - 1 and grid[row+1][col] == "#" and grid[row][col+1] == "#" and grid[row+1][col+1] == "#"):
                continue  # Skip this placement to avoid 2x2 block

            place_symmetrically(row, col)
            placed_count += 2 if (row,col) != (height-1-row,width-1-col) else 1

    # If not enough squares placed, don't infinitely recurse; log and return the best attempt.
    if placed_count < num_black_squares:
        logging.warning(f"Could only place {placed_count} of {num_black_squares} black squares.")
    return grid


def generate_grid(width: int, height: int, black_square_ratio: float, manual_grid: Optional[str] = None, grid_file: Optional[str] = None) -> List[List[str]]:
    """Generates grid using specified method, with fallback to random generation."""
    if manual_grid:
        grid = generate_grid_from_string(manual_grid)
        if grid:
            return grid
        logging.warning("Invalid manual grid. Generating random grid.")

    if grid_file:
        grid = generate_grid_from_file(grid_file)
        if grid:
            return grid
        logging.warning("Invalid grid file. Generating random grid.")

    return generate_grid_random(width, height, black_square_ratio)


def find_slots(grid: List[List[str]]) -> List[Tuple[int, int, str, int]]:
    """Identifies word slots (across and down) in the grid."""
    height, width = len(grid), len(grid[0])
    slots = []

    # Across slots
    for r in range(height):
        start = -1
        for c in range(width):
            if grid[r][c] == ".":
                if start == -1:
                    start = c
            elif start != -1:
                length = c - start
                if length >= MIN_WORD_LENGTH:
                    slots.append((r, start, "across", length))
                start = -1
        if start != -1 and width - start >= MIN_WORD_LENGTH:
            slots.append((r, start, "across", width - start))

    # Down slots
    for c in range(width):
        start = -1
        for r in range(height):
            if grid[r][c] == ".":
                if start == -1:
                    start = r
            elif start != -1:
                length = r - start
                if length >= MIN_WORD_LENGTH:
                    slots.append((start, c, "down", length))
                start = -1
        if start != -1 and height - start >= MIN_WORD_LENGTH:
            slots.append((start, c, "down", height - start))
    return slots

def is_valid_placement(grid: List[List[str]], word: str, row: int, col: int, direction: str) -> bool:
    """Checks if word can be placed at given location and direction."""
    length = len(word)
    if direction == "across":
        if col + length > len(grid[0]):
            return False
        for i in range(length):
            if grid[row][col + i] != "." and grid[row][col + i] != word[i]:
                return False
    else:  # down
        if row + length > len(grid):
            return False
        for i in range(length):
            if grid[row + i][col] != "." and grid[row + i][col] != word[i]:
                return False
    return True

def _is_valid_cached(grid: List[List[str]], word: str, row: int, col: int, direction: str) -> bool:
    """Cached version of is_valid_placement."""
    key = f"{word}:{row}:{col}:{direction}"
    with cache_lock:
        if key not in placement_cache:
            placement_cache[key] = is_valid_placement(grid, word, row, col, direction)
        return placement_cache[key]


def place_word(grid: List[List[str]], word: str, row: int, col: int, direction: str) -> List[List[str]]:
    """Places a word onto a *copy* of the grid."""
    new_grid = [row[:] for row in grid]  # Deep copy for safety
    for i, letter in enumerate(word):
        if direction == "across":
            new_grid[row][col + i] = letter
        else:
            new_grid[row + i][col] = letter
    return new_grid

def remove_word(grid: List[List[str]], word: str, row: int, col: int, direction: str) -> List[List[str]]:
    """Removes a word from a *copy* of the grid."""
    new_grid = [row[:] for row in grid]  # Deep copy for safety
    for i in range(len(word)):
        if direction == "across":
            if new_grid[row][col+i] == word[i]: #Only remove if it is the placed word
                new_grid[row][col + i] = "."
        else:
            if new_grid[row+i][col] == word[i]:
                new_grid[row + i][col] = "."

    return new_grid



def check_all_letters_connected(grid: List[List[str]], placed_words: List[Tuple[str, int, int, str]]) -> bool:
    """Checks if all placed letters are part of both across and down words."""
    if not placed_words:
        return True

    letter_positions = set()
    for word, row, col, direction in placed_words:
        for i in range(len(word)):
            if direction == "across":
                letter_positions.add((row, col + i))
            else:
                letter_positions.add((row + i, col))

    for row, col in letter_positions:
        in_across = any(
            row == r and col >= c and col < c + len(w)
            for w, r, c, d in placed_words if d == "across"
        )
        in_down = any(
            col == c and row >= r and row < r + len(w)
            for w, r, c, d in placed_words if d == "down"
        )
        if not (in_across and in_down):
            return False  # Letter not part of both across and down words

    return True  # All letters connected

def _validate_remaining_slots(grid: List[List[str]], slots: List[Tuple[int, int, str, int]], words_by_length: Dict[int, List[str]]) -> bool:
    """Checks if, for all remaining slots, at least one valid word exists (forward checking)."""
    for row, col, direction, length in slots:
        if not any(_is_valid_cached(grid, word, row, col, direction) for word in words_by_length.get(length, [])):
            return False  # No valid words for this slot
    return True

def calculate_intersection_score(grid:List[List[str]], word:str, row:int, col:int, direction:str, placed_words: List[Tuple[str, int, int, str]]) -> float:
    """Calculates an intersection score, rewarding more intersections and rarer letters."""

    intersections = 0
    score = 0.0
    length = len(word)

    for placed_word, p_row, p_col, p_dir in placed_words:
        if direction == "across" and p_dir == "down":
            if p_col >= col and p_col < col + length:  # Potential intersection
                if row >= p_row and row < p_row + len(placed_word):
                    intersections +=1
                    score += 1
        elif direction == "down" and p_dir == "across":
            if p_row >= row and p_row < row+length:
                if col >= p_col and col < p_col + len(placed_word):
                    intersections += 1
                    score +=1

    return score + intersections * 0.5 # intersections have additional score



def get_slot_score(
    grid: List[List[str]],
    slot: Tuple[int, int, str, int],
    words_by_length: Dict[int, List[str]],
    placed_words: List[Tuple[str,int,int,str]]
) -> float:
    """Scores a slot based on constrainedness and potential for good intersections."""
    row, col, direction, length = slot
    possible_words = sum(1 for w in words_by_length.get(length, []) if _is_valid_cached(grid, w, row, col, direction))
    if possible_words == 0:
        return -1.0  # No valid words, very bad slot

    # Calculate a constrainedness score (fewer options = higher priority)
    constrainedness_score = 1.0 / possible_words

    # Calculate intersection score by simulating placements
    intersection_potential = 0.0
    for word in words_by_length.get(length, []):
        if _is_valid_cached(grid,word,row,col,direction):
            intersection_potential += calculate_intersection_score(grid, word, row, col, direction, placed_words)

    return constrainedness_score + intersection_potential * 0.1 # intersection potential is less important.


# --- Word Selection (Recursive, with Backtracking and Parallelism) ---

class CrosswordStats:
    """Tracks statistics about the crossword generation process."""
    def __init__(self):
        self.attempts = 0
        self.backtracks = 0
        self.words_tried = 0
        self.successful_placements = 0
        self.failed_placements = 0
        self.time_spent = 0.0
        self.start_time = time.time()
        self.slot_fill_order: List[Tuple[int, int, str]] = [] # Log the order
        self.definition_failures = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def update_time(self):
        self.time_spent = time.time() - self.start_time

    def get_summary(self) -> str:
        """Returns a formatted summary of the statistics."""
        self.update_time()
        summary = (
            "📊 Crossword Generation Stats:\n"
            f"├── Attempts: {self.attempts}\n"
            f"├── Backtracks: {self.backtracks}\n"
            f"├── Words Tried: {self.words_tried}\n"
            f"├── Successful Placements: {self.successful_placements}\n"
            f"├── Failed Placements: {self.failed_placements}\n"
            f"├── Definition Failures: {self.definition_failures}\n"
            f"├── Placement Cache Hits: {self.cache_hits}\n"
            f"├── Placement Cache Misses: {self.cache_misses}\n"
            f"├── Success Rate: {self.successful_placements / max(1, self.words_tried) * 100:.2f}%\n"
            f"└── Time Spent: {self.time_spent:.2f}s\n"
            f"└── Slots Filled Order: {self.slot_fill_order}"
        )
        return summary

stats = CrosswordStats() # Global stats object

def validate_placement(grid: List[List[str]], slot: Tuple[int, int, str, int], 
                      word: str, remaining_slots: List[Tuple[int, int, str, int]], 
                      words_by_length: Dict[int, List[str]]) -> bool:
    """Validates if a word placement is valid and doesn't create impossible situations."""
    row, col, direction, _ = slot
    
    # Check basic placement validity
    if not _is_valid_cached(grid, word, row, col, direction):
        return False
        
    # Check if remaining slots are still fillable (forward checking)
    temp_grid = place_word(grid, word, row, col, direction)
    return _validate_remaining_slots(temp_grid, remaining_slots, words_by_length)

def make_placement(grid: List[List[str]], slot: Tuple[int, int, str, int], 
                  word: str, placed_words: List[Tuple[str, int, int, str]]) -> Tuple[List[List[str]], List[Tuple[str, int, int, str]]]:
    """Places a word and updates the placed_words list."""
    row, col, direction, _ = slot
    new_grid = place_word(grid, word, row, col, direction)
    new_placed_words = placed_words + [(word, row, col, direction)]
    
    stats.successful_placements += 1
    stats.words_tried += 1
    stats.slot_fill_order.append((row, col, direction))
    
    return new_grid, new_placed_words

def handle_backtrack(slot: Tuple[int, int, str, int]) -> None:
    """Handles statistics and cleanup during backtracking."""
    row, col, direction = slot[0], slot[1], slot[2]
    stats.backtracks += 1
    stats.failed_placements += 1
    if stats.slot_fill_order:  # Check if there's anything to pop
        stats.slot_fill_order.pop()

def try_slot(
    grid: List[List[str]],
    slot: Tuple[int, int, str, int],
    word: str,
    remaining_slots: List[Tuple[int, int, str, int]],
    words_by_length: Dict[int, List[str]],
    word_frequencies: Dict[str, float],
    placed_words: List[Tuple[str, int, int, str]],
    progress: Progress,
    task: TaskID,
    difficulty: str = DEFAULT_DIFFICULTY
) -> Tuple[Optional[List[List[str]]], Optional[List[Tuple[str, int, int, str]]]]:
    """Attempts to place a word in a slot and continues recursively (without threading)."""
    
    # Validate placement
    if not validate_placement(grid, slot, word, remaining_slots, words_by_length):
        return None, None
        
    # Make placement
    new_grid, new_placed_words = make_placement(grid, slot, word, placed_words)
    
    # Call select_words_recursive directly (no threading)
    result = select_words_recursive(
        new_grid,
        remaining_slots,
        words_by_length,
        word_frequencies,
        new_placed_words,
        progress,
        task,
        difficulty,
        None  # Pass None for executor to indicate no threading
    )
    
    if result[0] is not None:
        return result
        
    # Handle backtracking
    handle_backtrack(slot)
    return None, None

def adjust_black_squares_ratio(grid: List[List[str]], target_word_count: int, 
                             current_ratio: float) -> float:
    """Dynamically adjusts black square ratio based on word count."""
    slots = find_slots(grid)
    current_word_count = len(slots)
    
    if current_word_count < target_word_count:
        return max(0.05, current_ratio - 0.02)  # Decrease black squares
    elif current_word_count > target_word_count * 1.2:  # 20% tolerance
        return min(0.30, current_ratio + 0.02)  # Increase black squares
    return current_ratio

def select_words_recursive(
    grid: List[List[str]],
    slots: List[Tuple[int, int, str, int]],
    words_by_length: Dict[int, List[str]],
    word_frequencies: Dict[str, float],
    placed_words: List[Tuple[str, int, int, str]],
    progress: Progress,
    task: TaskID,
    difficulty: str = DEFAULT_DIFFICULTY,
    executor: Optional[ThreadPoolExecutor] = None
) -> Tuple[Optional[List[List[str]]], Optional[List[Tuple[str, int, int, str]]]]:
    """Recursively selects and places words, with controlled threading."""
    stats.attempts += 1
    stats.update_time()
    
    if stats.time_spent > DEFAULT_TIMEOUT:
        return None, None
        
    if not slots:
        if check_all_letters_connected(grid, placed_words):
            return grid, placed_words
        return None, None
    
    # Score and sort slots
    scored_slots = []
    for slot in slots:
        base_score = get_slot_score(grid, slot, words_by_length, placed_words)
        location_bonus = get_location_score(grid, slot)
        total_score = base_score + location_bonus
        scored_slots.append((total_score, slot))
    
    scored_slots.sort(key=lambda x: x[0], reverse=True)
    
    # Process top slots
    for score, slot in scored_slots[:DEFAULT_BEAM_WIDTH]:
        row, col, direction, length = slot
        remaining_slots = [s for s in slots if s != slot]
        
        # Score words considering difficulty
        word_scores = []
        freq_weight = WORD_FREQUENCY_WEIGHTS[difficulty]
        
        for word in words_by_length.get(length, []):
            if _is_valid_cached(grid, word, row, col, direction):
                intersection_score = calculate_intersection_score(grid, word, row, col, direction, placed_words)
                frequency_score = calculate_word_frequency(word, word_frequencies)
                word_score = (intersection_score * (1 - freq_weight) + 
                            (1 - frequency_score) * freq_weight)
                word_scores.append((word_score, word))
        
        word_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Use threading only at the top level
        if executor is not None:
            futures = []
            for _, word in word_scores[:DEFAULT_MAX_BACKTRACK]:
                future = executor.submit(
                    try_slot, grid, slot, word, remaining_slots,
                    words_by_length, word_frequencies, placed_words,
                    progress, task, difficulty
                )
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result[0] is not None:
                        return result
                except Exception as e:
                    logging.warning(f"Error in word placement thread: {e}")
                    continue
        else:
            # Sequential processing for recursive calls
            for _, word in word_scores[:DEFAULT_MAX_BACKTRACK]:
                result = try_slot(
                    grid, slot, word, remaining_slots,
                    words_by_length, word_frequencies, placed_words,
                    progress, task, difficulty
                )
                if result[0] is not None:
                    return result
    
    return None, None

def get_location_score(grid: List[List[str]], slot: Tuple[int, int, str, int]) -> float:
    """Calculates a score based on slot location and shape."""
    row, col, direction, length = slot
    height, width = len(grid), len(grid[0])
    
    # Center bonus (slots closer to center get higher scores)
    center_row, center_col = height // 2, width // 2
    distance_to_center = abs(row - center_row) + abs(col - center_col)
    center_bonus = 1.0 - (distance_to_center / (height + width))
    
    # Length bonus (longer words get slightly higher priority)
    length_bonus = length / max(height, width)
    
    # Intersection potential (slots that cross more other slots get higher scores)
    crossing_slots = 0
    if direction == "across":
        for i in range(length):
            if any(grid[r][col + i] == "." for r in range(height)):
                crossing_slots += 1
    else:
        for i in range(length):
            if any(grid[row + i][c] == "." for c in range(width)):
                crossing_slots += 1
                
    intersection_bonus = crossing_slots / length
    
    return (center_bonus * 0.4 + length_bonus * 0.3 + intersection_bonus * 0.3)

def select_words(
        grid: List[List[str]],
        slots: List[Tuple[int, int, str, int]],
        words_by_length: Dict[int, List[str]],
        word_frequencies: Dict[str, float],  # Pass word frequencies
        progress: Progress,
        task: TaskID
) -> Tuple[Optional[List[List[str]]], Optional[List[Tuple[str, int, int, str]]]]:
    """Initializes word selection with a single thread pool."""
    global stats
    stats = CrosswordStats()
    stats.start_time = time.time()
    
    initial_placed_words: List[Tuple[str, int, int, str]] = []
    
    # Create a single thread pool for top-level parallelization
    with ThreadPoolExecutor(max_workers=MAX_THREAD_POOL_SIZE) as executor:
        return select_words_recursive(
            grid, slots, words_by_length, word_frequencies,
            initial_placed_words, progress, task,
            DEFAULT_DIFFICULTY, executor
        )

# --- Definition and Crossword Generation ---
def order_cell_numbers(slots: List[Tuple[int, int, str, int]]) -> Dict[Tuple[int, int, str], int]:
    """Orders cell numbers for crossword clues (top-to-bottom, left-to-right)."""
    numbered_cells = set()  # Track cells that have been numbered
    cell_numbers: Dict[Tuple[int, int, str], int] = {}
    next_number = 1

    # Sort slots: first by row, then by column, then by direction (across before down)
    sorted_slots = sorted(slots, key=lambda x: (x[0], x[1], 0 if x[2] == "across" else 1))

    for row, col, direction, length in sorted_slots:
        if (row, col) not in numbered_cells:  # Only number if this cell starts a word
            cell_numbers[(row, col, direction)] = next_number
            numbered_cells.add((row, col))
            next_number += 1

    return cell_numbers

def generate_definitions(placed_words: List[Tuple[str, int, int, str]], llm: ChatOpenAI, language: str) -> Dict[str, Dict[int, str]]:
    """Generates definitions for placed words, with proper numbering."""
    definitions = {"across": {}, "down": {}}

    # First, order all cell numbers
    slots = [(row, col, direction, len(word)) for word, row, col, direction in placed_words]
    cell_numbers = order_cell_numbers(slots)

    with Progress() as progress:
        task = progress.add_task("[blue]Generating Definitions...", total=len(placed_words))
        with ThreadPoolExecutor() as executor:
            futures = []
            for word, row, col, direction in placed_words:
                future = executor.submit(generate_definition_langchain, llm, word, language)
                futures.append((future, word, row, col, direction)) #Store all data for easier processing

            for future, word, row, col, direction in futures:
                try:
                    definition = future.result()
                    number = cell_numbers.get((row, col, direction))
                    if number:
                        definitions[direction][f"{number}. {word}"] = definition
                except Exception as e:
                    stats.definition_failures += 1 #Increment failures
                    logging.error(f"Error retrieving definition for {word}: {e}")
                progress.update(task, advance=1)
    return definitions

def create_html(grid: List[List[str]], placed_words: List[Tuple[str, int, int, str]], definitions: Dict[str, Dict[int, str]], output_filename: str):
    """Generates the interactive HTML file for the crossword."""
    try:
        with open("template.html", "r", encoding="utf-8") as template_file: #Load from external file
            template = template_file.read()

        # Create grid HTML
        grid_html = '<table class="crossword-grid">'
        for row_index, row in enumerate(grid):
            grid_html += "<tr>"
            for col_index, cell in enumerate(row):
                if cell == "#":
                    grid_html += '<td class="black"></td>'
                else:
                    # Find the word (if any) that occupies this cell
                    word_info = None
                    for word, word_row, word_col, direction in placed_words:
                        if direction == "across" and row_index == word_row and word_col <= col_index < word_col + len(word):
                            word_info = (word, word_row, word_col, direction, col_index - word_col)
                            break
                        elif direction == "down" and col_index == word_col and word_row <= row_index < word_row + len(word):
                            word_info = (word, word_row, word_col, direction, row_index - word_row)
                            break

                    if word_info:  # Cell is part of a word
                        word, word_row, word_col, direction, index_in_word = word_info
                        cell_id = f"{word_row}-{word_col}-{direction}"
                        if index_in_word == 0:  # First letter of the word
                            # Get the correct number
                            slots = [(word, row, col, direction) for word, row, col, direction in placed_words]
                            cell_numbers = order_cell_numbers(slots)

                            number = cell_numbers.get((word_row, word_col, direction), "")  # Get cell number
                            grid_html += (
                                f'<td class="white" id="{cell_id}">'
                                f'<div class="cell-container">'
                                f'<span class="number">{number}</span>'
                                f'<input type="text" maxlength="1" class="letter" data-row="{row_index}" data-col="{col_index}" data-direction="{direction}">'
                                f'</div>'
                                f"</td>"
                            )

                        else:
                            grid_html += (
                                f'<td class="white" id="{cell_id}-{index_in_word}">'
                                 f'<div class="cell-container">'
                                f'<input type="text" maxlength="1" class="letter" data-row="{row_index}" data-col="{col_index}" data-direction="{direction}">'
                                 f'</div>'
                                f"</td>"
                            )
                    else:
                        grid_html += '<td class="white"></td>'  # Empty cell
            grid_html += "</tr>"
        grid_html += "</table>"

        # Create definitions HTML
        definitions_html = '<div class="definitions">'
        for direction, clues in definitions.items():
            definitions_html += f'<h3>{direction.capitalize()}</h3><ol>'
            for clue, definition in clues.items():
                definitions_html += f'<li><span class="clue-number">{clue.split(".")[0]}.</span> {definition}</li>'
            definitions_html += '</ol>'
        definitions_html += '</div>'

        # Combine into the template
        final_html = template.format(grid_html=grid_html, definitions_html=definitions_html)

        with open(output_filename, "w", encoding="utf-8") as output_file:
            output_file.write(final_html)

    except FileNotFoundError:
        logging.error("Could not find template.html. Please make sure it exists in the same directory.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred while generating HTML: {e}")
        sys.exit(1)


def main():
    """Main function to parse arguments and run crossword generation."""
    parser = argparse.ArgumentParser(description="Generates an interactive crossword puzzle.")
    parser.add_argument("--width", type=int, default=DEFAULT_GRID_WIDTH, help="Width of grid (columns).")
    parser.add_argument("--height", type=int, default=DEFAULT_GRID_HEIGHT, help="Height of grid (rows).")
    parser.add_argument("--black_squares", type=float, default=DEFAULT_BLACK_SQUARE_RATIO, help="Approximate % of black squares (0.0 to 1.0).")
    parser.add_argument("--manual_grid", type=str, default=None, help="Manually specify grid ('.'=white, '#'=black).")
    parser.add_argument("--grid_file", type=str, default=None, help="Path to file with grid layout.")
    parser.add_argument("--lm_studio_url", type=str, default=DEFAULT_LM_STUDIO_URL, help="LM Studio server URL.")
    parser.add_argument("--words_file", type=str, default=DEFAULT_WORDS_FILE, help="Path to words file (one word per line).")
    parser.add_argument("--output_filename", type=str, default=DEFAULT_OUTPUT_FILENAME, help="Output HTML filename.")
    parser.add_argument("--max_attempts", type=int, default=DEFAULT_MAX_ATTEMPTS, help="Max attempts to place a word.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout for word selection (seconds).")
    parser.add_argument("--llm_timeout", type=int, default=DEFAULT_LLM_TIMEOUT, help="Timeout for LLM requests (seconds).")
    parser.add_argument("--llm_max_tokens", type=int, default=DEFAULT_LLM_MAX_TOKENS, help="Max tokens for LLM responses.")
    parser.add_argument("--language", type=str, default=DEFAULT_LANGUAGE, help="Language for definitions.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name to use for definitions.")
    parser.add_argument("--max_grid_iterations", type=int, default=5, help="Maximum number of attempts to generate a complete grid.") #Added to the args

    args = parser.parse_args()

    # --- Input Validation (Comprehensive) ---
    if not all(isinstance(arg, int) and arg > 0 for arg in [args.width, args.height, args.max_attempts, args.timeout, args.llm_timeout, args.llm_max_tokens, args.max_grid_iterations]):
        logging.error("Width, height, max_attempts, timeout, llm_timeout, llm_max_tokens and max_grid_iterations must be positive integers.")
        sys.exit(1)
    if not 0.0 <= args.black_squares <= 1.0:
        logging.error("black_squares must be between 0.0 and 1.0.")
        sys.exit(1)
    if args.manual_grid and args.grid_file:
        logging.error("Cannot specify both --manual_grid and --grid_file.")
        sys.exit(1)

    # --- Load Words and Calculate Frequencies ---
    words_by_length, word_frequencies = load_words(args.words_file)
    if not words_by_length:
        logging.error("No valid words found in the word file.")
        sys.exit(1)

    # --- Setup LLM ---
    llm = setup_langchain_llm(args.lm_studio_url, args.llm_timeout, args.llm_max_tokens, args.model)

    # --- Main Generation Loop ---
    console = Console()  # Initialize Rich Console
    for attempt in range(args.max_grid_iterations):
        console.print(f"\n[bold blue]Attempting to generate crossword (Attempt {attempt + 1}/{args.max_grid_iterations})[/]")
        grid = generate_grid(args.width, args.height, args.black_squares, args.manual_grid, args.grid_file)

        if not is_valid_grid(grid): #Validate grid
            console.print("[red]Invalid grid generated. Retrying...[/]")
            continue

        console.print("[green]Initial Grid:[/]")
        print_grid(grid, console=console)

        slots = find_slots(grid)
        if not slots:
             console.print("[red]No valid slots found in the grid. Retrying...[/]")
             continue

        with Progress() as progress:
            task = progress.add_task("[cyan]Selecting words and generating crossword...", total=None)  # Indeterminate progress
            filled_grid, placed_words = select_words(grid, slots, words_by_length, word_frequencies, progress, task)
            progress.update(task,completed=100) #Complete the task

        if filled_grid is not None:
            console.print("[green]Crossword filled successfully![/]")
            print_grid(filled_grid,placed_words, console)

            definitions = generate_definitions(placed_words, llm, args.language)
            create_html(filled_grid, placed_words, definitions, args.output_filename)
            console.print(f"[green]Crossword puzzle saved to: {args.output_filename}[/]")
            console.print(stats.get_summary()) # Show stats
            break  # Exit loop on success
        else:
            console.print("[yellow]Failed to fill the grid completely. Retrying with a new grid...[/]")
            console.print(stats.get_summary()) #Show stats
    else:
        console.print("[red]Failed to generate a complete crossword after multiple attempts.[/]")

if __name__ == "__main__":
    main()