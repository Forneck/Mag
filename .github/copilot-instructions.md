# MAG: Sistema Multiagente Gemini - Development Instructions

**ALWAYS follow these instructions first and only fallback to additional search and context gathering if the information in the instructions is incomplete or found to be in error.**

MAG (Multi-Agent Gemini) is a Python-based multi-agent system that uses Google's Gemini AI models to automate complex tasks through a specialized agent architecture. The system employs a TaskManager to decompose high-level objectives into manageable subtasks, executed by specialized Worker agents, and validated by a Validator agent.

## Environment Setup and Dependencies

### Prerequisites
- Python 3.12 or higher (confirmed working with Python 3.12.3)
- Valid Google Gemini API key
- Internet connectivity for package installation and API calls

### Required Dependencies Installation
**CRITICAL: Network connectivity issues are common during dependency installation. Be patient and retry if needed. Installation may take 2-5 minutes.**

Install the required Python packages:
```bash
pip3 install google-genai pillow
```

**IMPORTANT DEPENDENCY NOTE**: The application requires `google-genai` (NOT `google-generativeai`). This is Google's newer experimental Gemini API package (version 1.30.0+) that provides the `genai.Client` and `genai.types` interfaces used by the code.

**KNOWN ISSUE**: Network timeouts are frequent during installation. If installation fails:
```bash
# Retry with extended timeout (NEVER CANCEL - wait up to 5 minutes)
pip3 install --timeout 300 --retries 3 google-genai pillow

# If still failing, try individual packages
pip3 install --timeout 300 pillow
pip3 install --timeout 300 google-genai
```

**VALIDATION**: After installation, verify dependencies work:
```bash
python3 -c "from google import genai; print('✓ google-genai import works')"
python3 -c "from PIL import Image; print('✓ Pillow import works')"
```

If the first command fails with `ImportError: cannot import name 'genai' from 'google'`, the `google-genai` package was not installed correctly.

### Environment Variables
Set your Google Gemini API key:
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

**CRITICAL**: The application will exit immediately if `GEMINI_API_KEY` is not set.

## Repository Structure and Key Files

### Main Files
- `mag.py` - Main application file (346 lines) containing all classes and logic
- `README.md` - Comprehensive documentation in Portuguese
- `LICENSE` - MIT license file

### Runtime Directories (Auto-created)
The application automatically creates these directories:
- `gemini_agent_logs/` - Detailed execution logs with timestamps
- `gemini_uploaded_files_cache/` - Metadata cache for uploaded files
- `gemini_temp_artifacts/` - Temporary artifacts during execution (cleaned automatically)
- `gemini_final_outputs/` - Final approved outputs organized by timestamp

## Running the Application

### Basic Execution
```bash
python3 mag.py
```

**TIMING: Application startup takes 5-10 seconds. NEVER CANCEL during initialization.**

### Application Flow
1. **Cache Management**: Optional cleanup of local cache and API files
2. **File Upload**: Optional file upload with wildcard support (e.g., `*.txt`)
3. **Goal Definition**: Interactive goal input (type 'FIM' to finish)
4. **Task Planning**: TaskManager decomposes goal into subtasks
5. **User Approval**: Manual approval of generated task plan
6. **Task Execution**: Sequential execution by Worker agents
7. **Validation**: Final validation and approval cycle

### Interactive Elements
- File upload prompt: Supports wildcards like `*.txt`, `*.pdf`
- Goal definition: Multi-line input terminated by typing 'FIM'
- Plan approval: Type 's' to approve, anything else to reject
- Validation menu: `[A]provar`, `[F]eedback`, `[S]air` options

## Key Classes and Architecture

### TaskManager
- **Purpose**: Central orchestrator and task decomposer
- **Key Methods**:
  - `decompose_goal()`: Breaks down main goal into JSON task list
  - `run_workflow()`: Manages complete execution flow
- **Configuration**: Uses `gemini-2.5-flash-preview-05-20` model

### Worker
- **Purpose**: Executes individual subtasks
- **Key Methods**:
  - `execute_task()`: Processes single task with context
- **Capabilities**: Text processing, code generation, tool calling

### Available Tools
- `save_file()`: Saves text content to files in output directory
- `generate_image()`: Creates images using `gemini-2.0-flash-preview-image-generation`

### Configuration Constants (in mag.py)
```python
GEMINI_TEXT_MODEL_NAME = "gemini-2.5-flash-preview-05-20"
GEMINI_IMAGE_MODEL_NAME = "gemini-2.0-flash-preview-image-generation"
MAX_API_RETRIES = 3
INITIAL_RETRY_DELAY_SECONDS = 5
RETRY_BACKOFF_FACTOR = 2
```

## Testing and Validation

### Pre-flight Checks
Before running the application, verify setup:
```bash
# Check Python version
python3 --version

# Verify dependencies
python3 -c "from google import genai; print('✓ google-genai import works')"
python3 -c "from PIL import Image; print('✓ Pillow import works')"

# Test API key environment variable
echo $GEMINI_API_KEY
```

### Manual Testing Scenarios
**ALWAYS test these scenarios after making changes:**

1. **Basic Goal Processing**:
   - Start application
   - Skip file uploads (press Enter)
   - Enter simple goal like "Create a hello world Python script"
   - Type 'FIM' to finish
   - Approve the generated plan
   - Verify file creation in `gemini_final_outputs/`

2. **File Upload Testing**:
   - Create test files: `echo "test content" > test.txt`
   - Start application 
   - Upload `*.txt` files
   - Enter goal utilizing uploaded content
   - Verify context usage in task execution

3. **Image Generation Testing**:
   - Enter goal requiring image creation
   - Verify images are saved to output directory
   - Check PNG file format and content

### Expected Runtime Behavior
- **Startup**: 5-10 seconds for API initialization - NEVER CANCEL
- **Task decomposition**: 10-30 seconds depending on complexity - NEVER CANCEL
- **Task execution**: Variable (10 seconds to several minutes per task) - NEVER CANCEL
- **Image generation**: 30-60 seconds per image - NEVER CANCEL
- **File operations**: Near-instantaneous

**CRITICAL TIMING NOTES:**
- **NEVER CANCEL** any operation, even if it seems stuck
- Set timeouts to at least 120 seconds for any automated testing
- The system uses exponential backoff for API retries (can extend operation time)
- First run may take longer due to API client initialization

## Common Issues and Solutions

### Import Errors
```
ImportError: cannot import name 'genai' from 'google'
```
**Solution**: Install `google-genai` (not `google-generativeai`):
```bash
pip3 install google-genai
```

### API Key Errors
```
ValueError: A variável de ambiente GEMINI_API_KEY não está definida
```
**Solution**: Set environment variable before running:
```bash
export GEMINI_API_KEY="your_key_here"
python3 mag.py
```

### Network Timeout During Installation
**Solution**: Retry with extended timeout:
```bash
pip3 install --timeout 300 --retries 3 google-genai pillow
```

### Empty Goal Error
**Solution**: Always provide meaningful goal text before typing 'FIM'

## Development Guidelines

### Making Changes
- **ALWAYS** backup your API key and test files before changes
- Test changes with simple goals first before complex scenarios  
- Monitor log files in `gemini_agent_logs/` for debugging
- Check output directory structure after each test

### Code Structure Navigation
- **Lines 1-74**: Imports, configuration, and API setup
- **Lines 75-120**: Tool functions (save_file, generate_image)
- **Lines 121-200**: Utility functions and API calling logic
- **Lines 201-270**: TaskManager class
- **Lines 271-327**: Worker class  
- **Lines 328-346**: Main execution logic

### Performance Considerations
- Each API call has built-in retry logic (3 attempts)
- Exponential backoff prevents API rate limiting
- File uploads are cached to avoid re-uploading
- Temporary artifacts are auto-cleaned

## Validation Requirements

**Before considering changes complete, ALWAYS:**
1. Run basic goal processing scenario successfully
2. Verify log file creation with timestamp in `gemini_agent_logs/`
3. Check output directory contains expected artifacts in `gemini_final_outputs/`
4. Test file upload functionality with sample files
5. Confirm error handling with invalid/missing API key
6. Validate cleanup of temporary directories after completion

## Additional Development Information

### File Patterns and Wildcards
The application supports these file upload patterns:
- `*.txt` - All text files
- `*.py` - All Python files  
- `*.pdf` - All PDF files
- `data/*.csv` - CSV files in data directory
- Press Enter without input to skip file uploads

### Logging and Debugging
- All operations are logged to `gemini_agent_logs/agent_log_[timestamp].txt`
- Log includes API calls, responses, errors, and execution flow
- Monitor logs in real-time: `tail -f gemini_agent_logs/agent_log_*.txt`
- Logs are essential for debugging API or network issues

### Directory Structure After Execution
```
./
├── mag.py
├── README.md
├── gemini_agent_logs/
│   └── agent_log_YYYYMMDD_HHMMSS.txt
├── gemini_uploaded_files_cache/
├── gemini_temp_artifacts/ (cleaned automatically)
├── gemini_final_outputs/
│   └── YYYYMMDD_HHMMSS/
│       ├── generated_files...
│       └── evaluation_report.md
└── test.txt (if created for testing)
```

### Limitations and Known Issues
1. **Dependency Installation**: Network timeouts are common - retry multiple times
2. **API Key Required**: Application cannot run in demo/offline mode
3. **Portuguese Interface**: Most user prompts and messages are in Portuguese
4. **Network Dependency**: Requires internet for API calls during execution
5. **No Build Step**: No compilation or build process - direct Python execution

**CRITICAL VALIDATION NOTE**: Due to network connectivity issues in the validation environment, the `google-genai` dependency installation could not be completed. However, all other aspects of the instructions have been validated, including:
- Python version compatibility (3.12.3)
- Pillow dependency installation and import
- File structure analysis (346 lines in mag.py)
- Code architecture understanding
- Directory creation patterns
- Basic file operations

## Security Notes
- Never commit API keys to source control
- Uploaded files are stored locally only during session
- Generated artifacts are saved locally in timestamped directories
- API communications use Google's standard authentication

**TIMING REMINDER: Be patient with all operations. The system is designed for reliability over speed.**onfirm error handling with invalid API key
6. Validate cleanup of temporary directories

## Security Notes
- Never commit API keys to source control
- Uploaded files are stored locally only during session
- Generated artifacts are saved locally in timestamped directories
- API communications use Google's standard authentication

**TIMING REMINDER: Be patient with all operations. The system is designed for reliability over speed.**