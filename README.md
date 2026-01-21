# Exploring RNNs: LSTM-based Python Code Generation

An experimental project investigating how LSTMs handle long-term dependencies through character-level Python code generation. This project uses code as a testbed because it naturally contains complex long-range dependencies like matching parentheses, consistent indentation, and nested structures.

## Motivation

Long-term dependencies pose a fundamental challenge for recurrent neural networks. While LSTMs were designed to address the vanishing gradient problem and capture longer-range dependencies, their practical limits remain an open question. 

**Why Python code generation?**

Python code provides an ideal benchmark for testing long-term dependency learning because it requires the model to:
- **Match opening and closing delimiters** (parentheses, brackets, braces) across potentially hundreds of characters
- **Maintain consistent indentation** across function and class bodies spanning multiple lines
- **Remember context** for variable names, function signatures, and control flow structures
- **Handle nested structures** like nested function calls, list comprehensions, and conditional blocks

These characteristics make code generation a challenging and revealing test case for LSTM capabilities.

## Long-Term Dependencies in Code

LSTMs must track several types of long-range dependencies when generating syntactically valid Python:

### 1. **Bracket Matching**
```python
def function(arg1, arg2, nested_call(
    another_function(x, y, z),
    more_args
)):  # Must close all parentheses correctly
```

### 2. **Indentation Consistency**
```python
def outer_function():
    # 4 spaces
    if condition:
        # 8 spaces
        for item in list:
            # 12 spaces - must maintain this depth
            process(item)
```

### 3. **String/Docstring Delimiters**
```python
def example():
    """
    This docstring spans multiple lines
    with various content in between
    """  # Must close with triple quotes
```

### 4. **Control Flow Structure**
```python
if condition:
    # ... many lines ...
else:
    # Model must remember we're in an if-else block
```

## Architecture

The model uses a character-level LSTM architecture with the following components:

```
Input (batch_size, seq_length)
    ↓
Embedding Layer (vocab_size → 64 dims)
    ↓
LSTM Layer 1 (256 units, return_sequences=True)
    ↓
Dropout (0.1)
    ↓
LSTM Layer 2 (256 units, return_sequences=True)
    ↓
Dropout (0.1)
    ↓
Dense Layer (vocab_size, L2 regularization)
    ↓
Output (character predictions)
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Sequence Length | 256 characters |
| Batch Size | 512 |
| Embedding Dimension | 64 |
| LSTM Units (per layer) | 256 |
| Number of LSTM Layers | 2 |
| Dropout Rate | 0.1 |
| Learning Rate | 1.8e-4 |
| Optimizer | Adam (clipnorm=1) |
| L2 Regularization | 1e-4 |
| Training Epochs | 300 |
| Precision | Mixed Float16 |

### Key Design Choices

- **Character-level tokenization**: Captures all syntactic details including whitespace and punctuation
- **Stateful inference**: Maintains hidden states across generation steps for consistent context
- **Dropout + L2 regularization**: Prevents overfitting on code patterns
- **Gradient clipping**: Stabilizes training on long sequences
- **Mixed precision training**: Enables larger batch sizes with limited GPU memory

## Dataset

- **Size**: ~165,000 lines of Python source code (~6 MB)
- **Tokenization**: Character-level (including all ASCII letters, digits, punctuation, whitespace)
- **Vocabulary**: ~95 unique characters
- **Split**: 95% training, 5% validation
- **Sequence windows**: 256 characters with 1-character shift for target

## Project Structure

```
exploring-RNNs/
├── README.md                    # This file
└── python-code-generator/
    ├── train.py                 # Training script with model definition
    ├── inference.py             # Stateful text generation
    ├── data.txt                 # Training corpus (~165k lines)
    ├── trained_model.keras      # Saved model checkpoint
    └── generated.txt            # Sample generated outputs
```

## Installation & Requirements

### Dependencies
```bash
pip install tensorflow numpy matplotlib
```

### Requirements
- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib (for training visualization)
- GPU with CUDA support (recommended)

### GPU Configuration
- Mixed precision (float16) enabled
- Adjust `memory_limit` in `train.py:17` for different GPU configurations

## Usage

### Training

```bash
cd python-code-generator
python train.py
```

The training script will:
1. Load and preprocess `data.txt`
2. Create character-level vocabulary mappings
3. Build and compile the LSTM model
4. Train for 300 epochs with validation
5. Save the best model to `model_checkpoint.keras`

Training progress includes loss and accuracy metrics for both training and validation sets.

### Inference (Text Generation)

```bash
python inference.py
```

You'll be prompted for:
- **Number of characters to generate**: How long the output should be (e.g., 1000)
- **Temperature**: Controls randomness (0.1 = conservative, 1.0 = creative)
- **Start string**: Initial prompt (e.g., "def function")

#### Temperature Parameter

Temperature controls the randomness of character sampling:
- **Low (0.1-0.3)**: More deterministic, follows common patterns, syntactically safer
- **Medium (0.4-0.7)**: Balanced creativity and structure
- **High (0.8-1.0)**: More creative but potentially less syntactically valid

### Stateful Generation

The inference model maintains LSTM hidden states across generation steps, allowing it to:
- Remember opening brackets and quotes
- Maintain indentation context
- Track variable and function names
- Preserve overall code structure

## Sample Outputs

Below is a sample of generated code (temperature=0.5, 1000 characters):

```python
import the context by an invalid target.
            # This is array of the start link of the first because the same context provided.
            # if the item of the second back to the {backends to return the non-dataset values and which match
            # description is set the array.
            if default_scores in self._namespaces:
                self._info._get_classe(result, attrs)
            else:
                self.module_path = task
            else:
                self._parse_active(
                    self._tags,
                    self.filter_state, self.services,
                    self._selected_timestamp.test_text, test_to_text, timestamp)

    def async_remove_module(self, context, context):
        """
        Return the test that it uses themes.

        :param str: Any of the resource string is used in the contents of the
            are of the report loader of the module.
        """
        if self.parent_id is not None:
            return self._get_input("index %s interface")
        except IndexError:
            # Comments about collections and strings
            if not is is None:
                try:
                    if line.split() == '1':
                        self._set_index(self._dist, value)

                return False

    def test_data_to_internal(self, bool):
        self._set_context_components()
        self.assertEqual(resultset, "--2000 * scheme)
        self.assertEqual(len(sub_socket, timeout))
        self.assertEqual(results['instanceed_id', '_limit_method'])
```

### Observations from Generated Code

**What the model does well:**
- Generates plausible function definitions with `def` keyword and colons
- Maintains basic indentation patterns within small blocks
- Creates docstrings with opening/closing triple quotes
- Produces variable names and method calls that look "Python-like"
- Generates syntactically valid assertions and comments
- Remembers to close some parentheses and brackets

**Where long-term dependencies break down:**
- Logical inconsistencies (e.g., `return` after `if` without proper `try` block)
- Variable scope errors (using undefined variables)
- Unmatched delimiters over longer distances
- Semantic incoherence (function behavior doesn't match docstrings)
- Indentation drift in deeply nested structures
- Invalid syntax combinations (e.g., `if not is is None`)

**Interesting behaviors:**
- Creates plausible-sounding method names (`_parse_active`, `test_data_to_internal`)
- Generates realistic comment patterns explaining code intent
- Mimics common Python idioms (list comprehensions, context managers)
- Produces test-like code with `assertEqual` patterns
- Exhibits vocabulary from the training domain (dataset, context, state, etc.)

## Analysis: LSTM Performance on Long-Term Dependencies

### Successes (Short-Range Dependencies)
The LSTM successfully learns patterns within ~10-50 characters:
- Function signatures with parameters
- Simple if-else blocks
- Single-line statements
- Basic indentation after colons

### Challenges (Long-Range Dependencies)
The model struggles with dependencies spanning >100 characters:
- Matching opening/closing brackets separated by multiple lines
- Maintaining consistent indentation across entire functions
- Remembering variable definitions from earlier in the sequence
- Coherent logical flow across function bodies

### Implications
This experiment demonstrates the practical limits of LSTM architectures for tasks requiring very long-range context. While the 256-unit LSTM cells can theoretically maintain information across sequences, in practice:
1. **Information degrades** over distances >100 characters
2. **Syntax is easier than semantics** - the model learns surface patterns better than logical structure
3. **Local coherence beats global coherence** - individual lines look valid, but overall structure may not

## References & Related Work

This project was inspired by research on:
- LSTM capabilities for long-term dependency learning
- Neural code generation and program synthesis
- Character-level language modeling
- Applications of RNNs to structured data (code, markup languages)

## License

This is an experimental and educational research project for exploring LSTM capabilities.

## Acknowledgments

Training data consists of open-source Python code. This project is for educational and research purposes only.
