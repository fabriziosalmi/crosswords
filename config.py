# config.py

# --- Constants ---
DEFAULT_GRID_WIDTH = 5
DEFAULT_GRID_HEIGHT = 4
DEFAULT_BLACK_SQUARE_RATIO = 0.17  # Classic NYT ratio
DEFAULT_LM_STUDIO_URL = "http://localhost:1234/v1"
DEFAULT_WORDS_FILE = "data/parole.txt"  # Replace with your word list
DEFAULT_OUTPUT_FILENAME = "docs/cruciverba.html"
DEFAULT_MAX_ATTEMPTS = 10000  # Attempts per word placement
DEFAULT_TIMEOUT = 180  # Overall timeout (seconds)
DEFAULT_LLM_TIMEOUT = 30
DEFAULT_LLM_MAX_TOKENS = 64
DEFAULT_LANGUAGE = "Italian"  # Or whichever language
DEFAULT_MODEL = "meta-llama-3.1-8b-instruct"  # A good, general-purpose open model
MAX_RECURSION_DEPTH = 1000  # Safety net
DEFAULT_BEAM_WIDTH = 100
DEFAULT_MAX_BACKTRACK = 3000
MIN_WORD_LENGTH = 3  # More standard minimum length
FORBIDDEN_PATTERNS = [
    r'\b{}\b',  # Whole word
    r'{}'.format,  # Substring
    r'(?i){}'.format,  # Case-insensitive
]
MAX_DEFINITION_ATTEMPTS = 3
DEFINITION_RETRY_DELAY = 2
DEFAULT_DIFFICULTY = "medium"  # easy, medium, hard
WORD_FREQUENCY_WEIGHTS = {
    "easy": 0.8,
    "medium": 0.5,
    "hard": 0.2,
}
MIN_WORD_COUNTS = {
    "easy": 2,
    "medium": 3,
    "hard": 5
}

MAX_THREAD_POOL_SIZE = 8  # Limit maximum number of threads


class Config:
    """Configuration class for crossword generation."""

    def __init__(self):
        # Grid settings
        self.grid_width = DEFAULT_GRID_WIDTH
        self.grid_height = DEFAULT_GRID_HEIGHT
        self.black_square_ratio = DEFAULT_BLACK_SQUARE_RATIO
        self.manual_grid = None
        self.grid_file = None

        # LLM settings
        self.lm_studio_url = DEFAULT_LM_STUDIO_URL
        self.model = DEFAULT_MODEL
        self.llm_timeout = DEFAULT_LLM_TIMEOUT
        self.llm_max_tokens = DEFAULT_LLM_MAX_TOKENS

        # Word list and output
        self.words_file = DEFAULT_WORDS_FILE
        self.output_filename = DEFAULT_OUTPUT_FILENAME

        # Algorithm settings
        self.max_attempts = DEFAULT_MAX_ATTEMPTS
        self.timeout = DEFAULT_TIMEOUT
        self.language = DEFAULT_LANGUAGE
        self.beam_width = DEFAULT_BEAM_WIDTH
        self.max_backtrack = DEFAULT_MAX_BACKTRACK
        self.min_word_length = MIN_WORD_LENGTH
        self.forbidden_patterns = FORBIDDEN_PATTERNS
        self.max_definition_attempts = MAX_DEFINITION_ATTEMPTS
        self.definition_retry_delay = DEFINITION_RETRY_DELAY
        self.difficulty = DEFAULT_DIFFICULTY
        self.word_frequency_weights = WORD_FREQUENCY_WEIGHTS
        self.min_word_counts = MIN_WORD_COUNTS
        self.max_thread_pool_size = MAX_THREAD_POOL_SIZE
        self.max_grid_iterations = 5 # Added from main() default

    def update_from_args(self, args):
        """Updates configuration from argparse arguments."""
        self.grid_width = args.width
        self.grid_height = args.height
        self.black_square_ratio = args.black_squares
        self.manual_grid = args.manual_grid
        self.grid_file = args.grid_file
        self.lm_studio_url = args.lm_studio_url
        self.words_file = args.words_file
        self.output_filename = args.output_filename
        self.max_attempts = args.max_attempts
        self.timeout = args.timeout
        self.llm_timeout = args.llm_timeout
        self.llm_max_tokens = args.llm_max_tokens
        self.language = args.language
        self.model = args.model
        self.difficulty = args.difficulty
        self.max_grid_iterations = args.max_grid_iterations