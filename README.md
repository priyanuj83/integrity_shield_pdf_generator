# IntegrityShield Enhanced Pipeline

## 🚀 **Advanced Question Paper Generation System**

This pipeline generates realistic academic question papers across 22 domains with intelligent academic level differentiation using advanced dataset selection from MMLU, MMLU-Pro, GSM8K, MBPP+, AI2-ARC, PubMedQA, and SQuAD datasets with **fixed question distribution**.

## 📊 **Current Performance**
- **✅ High Success Rate** (1074+ papers generated successfully in latest run)
- **📚 22 Domains** supported with realistic academic level mappings
- **🎓 Smart Academic Levels**: K-12, Undergraduate, Graduate (domain-appropriate)
- **📄 Perfect 40 Marks** per paper with fixed question type distribution
- **🔄 Full Dataset Integration**: MMLU, MMLU-Pro, GSM8K, MBPP+, AI2-ARC, PubMedQA, SQuAD working seamlessly
- **📊 Fixed Distribution**: All papers contain exactly 5 MCQ, 5 True/False, and 2 Long-form questions (40 marks total)
- **🔧 Comprehensive Error Handling**: Automatic fallback mechanisms and detailed logging
- **📈 Scalable**: Configurable papers per domain-level (default: 25, easily adjustable)
- **⚡ Parallel Processing**: Multi-threaded generation with configurable worker counts and PDF compilation limits
- **🔒 Thread-Safe**: Thread-safe random number generation for parallel execution

## 🏗️ **Project Structure**

```
integrity_shield-sample_pdfs/
├── main.py                    # Main pipeline (enhanced with hierarchical generation)
├── config.yaml               # Configuration with hierarchical mappings
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── data/                     # Dataset storage
│   ├── raw/                 # Raw dataset files
│   │   ├── mmlu_all.json
│   │   ├── mmlu_pro.json
│   │   ├── mbpp_plus.json
│   │   ├── ai2_arc.json
│   │   ├── pubmedqa.json
│   │   └── squad.json
│   ├── gsm_mcq/             # GSM8K MCQ data
│   ├── cache/                # LLM conversion cache
│   ├── metadata_hierarchical/ # Paper metadata
│   └── gold_labels_hierarchical/ # Answer keys
│
├── src/                     # Source code
│   ├── data_processing/     # Dataset loading and processing
│   ├── pdf_generation/      # LaTeX and PDF generation
│   └── utils/               # Configuration and logging utilities
│
├── output/                  # Generated papers (hierarchical structure)
│   ├── science/             # K-12 Science (AI2-ARC dataset)
│   │   └── k-12/
│   ├── religious_studies/   # K-12 Religious Studies
│   │   └── k-12/
│   ├── mathematics/
│   │   ├── k-12/
│   │   ├── undergraduate/
│   │   └── graduate/
│   ├── business_administration/
│   │   ├── undergraduate/
│   │   └── graduate/
│   ├── health_sciences/
│   │   ├── k-12/
│   │   ├── undergraduate/
│   │   └── graduate/
│   ├── constitutional_law/
│   │   ├── undergraduate/
│   │   └── graduate/
│   └── ... (all 22 domains)
│
├── docs/                    # Documentation
├── examples/                # Example files
└── logs/                    # Log files (integrity_shield.log)
```

## 🎯 **Supported Domains with Realistic Academic Levels**

### **🧪 K-12 Science (AI2-ARC Dataset)**
1. **Science** - K-12 general science using AI2-ARC dataset (MCQ + True/False)

### **📚 Full Academic Spectrum (K-12, Undergraduate, Graduate)**
2. **Mathematics** - Complete progression from basic to advanced
3. **Computer Science** - Programming basics to advanced CS
4. **Economics** - Economic concepts to graduate economics
5. **Health Sciences** - Health education to advanced health studies
6. **History** - Historical knowledge to graduate research
7. **Geography** - Geographic concepts to advanced geography

### **🎓 Undergraduate & Graduate Only (Realistic Advanced Domains)**
8. **Physics** - College-level to graduate physics (no K-12)
9. **Chemistry** - Undergraduate to graduate chemistry (no K-12)
10. **Biology** - Undergraduate to graduate biology (no K-12)
11. **Engineering** - Undergraduate to graduate engineering
12. **Psychology** - College-level to graduate psychology
13. **Philosophy** - Undergraduate to graduate philosophy
14. **Business Administration** - Business studies to graduate business
15. **Constitutional Law** - Legal studies to graduate law
16. **Sociology** - Undergraduate to graduate sociology
17. **Political Science** - Political studies to graduate level
18. **Machine Learning** - Advanced undergraduate to graduate ML
19. **Cybersecurity** - Advanced undergraduate to graduate cybersecurity

### **🏫 K-12 Only (School-Level Subjects)**
20. **Religious Studies** - K-12 religious education

### **🔬 Graduate Only (Highly Specialized)**
21. **Astronomy** - Graduate-level astronomy only

## 🚀 **Quick Start**

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up OpenAI API key (for LLM-based long-form question conversion):**
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```
   - The `.env` file is automatically loaded by the application
   - **Note:** Without the API key, the system will fall back to rule-based conversion (lower quality)

3. **Download datasets (first time only):**
   ```bash
   python main.py
   ```

4. **Generate papers (with cached datasets):**
   ```bash
   python main.py --skip-download
   ```

5. **Generate papers with verbose logging:**
   ```bash
   python main.py --skip-download --verbose
   ```

## 🔧 **Command Line Arguments**

```bash
python main.py [OPTIONS]
```

**Available Options:**
- `--config PATH` - Configuration file path (default: `config.yaml`)
- `--count N` - Number of documents to generate (default: 5, not used in hierarchical mode)
- `--skip-download` - Reuse cached datasets (skip download phase)
- `--refresh-data` - Force re-download even if cached
- `--seed N` - Random seed for reproducibility (default: 42)
- `--verbose` - Enable verbose (DEBUG level) logging
- `--workers N` - Number of parallel workers for question generation (default: 6, max: 20)
- `--max-compile-jobs N` - Maximum concurrent PDF compilations (default: 4, max: 8). Limits pdflatex processes to prevent resource exhaustion.

**Examples:**
```bash
# Generate all papers with default settings (6 workers, 4 concurrent PDF compilations)
python main.py --skip-download

# Generate with custom seed for reproducibility
python main.py --skip-download --seed 123

# Generate with more parallel workers (faster, but uses more resources)
python main.py --skip-download --workers 8

# Generate with more concurrent PDF compilations (faster PDF generation, but more resource-intensive)
python main.py --skip-download --max-compile-jobs 6

# Generate with verbose logging for debugging
python main.py --skip-download --verbose

# Force refresh datasets and generate
python main.py --refresh-data

# Custom parallelization settings (balanced for laptops)
python main.py --skip-download --workers 6 --max-compile-jobs 4
```

## 🔧 **Recent Fixes & Improvements**

### **✅ Latest Updates (December 2025)**
- **⚡ Parallel Processing**: Implemented multi-threaded paper generation with configurable worker counts
  - Default: 6 workers for question generation, 4 concurrent PDF compilations
  - Semaphore-based PDF compilation limiting to prevent resource exhaustion
  - Thread-safe random number generation using local `random.Random` instances
  - Significant performance improvement: ~3-5x faster generation times
- **🔒 Thread-Safety**: Fixed thread-safety issues with global `random` module
  - Replaced global `random` with local `random.Random` instances per `DatasetPool`
  - Unique seeds per (domain, level) combination for deterministic parallel execution
  - Prevents race conditions and corrupted random states
- **📐 LaTeX Rendering Fixes**: Comprehensive fixes for math equation rendering
  - Fixed double-escaping of LaTeX special characters
  - Proper math mode handling for exponents (`^` characters)
  - Added `amsmath` and `amssymb` packages to answer key templates
  - Fixed option numbering (was showing "4. 5. 6. 7." instead of "(a) (b) (c) (d)")
  - Text sanitization to remove control characters and normalize Unicode math symbols
- **Domain Name Mapping Fixed**: Added mappings for `business_administration`, `health_sciences`, and `constitutional_law` to enable paper generation
- **MMLU-Pro Empty Pool Handling**: Added comprehensive fallback mechanisms when MMLU-Pro dataset is filtered to empty list
- **Long-Form Question Fallback**: Automatic fallback to regular MMLU when MMLU-Pro pool is empty during long-form question generation
- **Enhanced Error Handling**: Improved error messages with actionable guidance for configuration issues
- **Comprehensive Logging**: Added detailed logging throughout the pipeline (question extraction, building, rendering)
- **Index Error Fixes**: Fixed "list index out of range" errors by adding validation for empty option lists
- **Text Rendering Fix**: Fixed missing spaces between words in generated papers using intelligent space insertion
- **MMLU Subject Name Fixes**: Fixed subject name mismatches (e.g., `computer science` → `college_computer_science`)

### **🎯 Key Technical Improvements**
- **Parallel Architecture**: ThreadPoolExecutor-based parallelization at (domain, level) granularity
  - Each worker generates all papers for one (domain, level) combination
  - Efficient resource utilization with configurable concurrency limits
  - Progress tracking and real-time status updates
- **Resource Management**: Semaphore-based PDF compilation limiting
  - Prevents "subprocess storm" from too many concurrent `pdflatex` processes
  - Configurable via `--max-compile-jobs` argument
  - Prevents system crashes and resource exhaustion
- **Dataset Pool Validation**: Early detection of empty datasets with clear warnings
- **Graceful Fallbacks**: Automatic dataset switching when primary source is unavailable
- **Question Validation**: Enhanced validation for MCQ options, choices, and answer indices
- **Logging Infrastructure**: Comprehensive logging saved to `logs/integrity_shield.log`
- **Error Recovery**: Better error handling with fallback mechanisms at multiple levels
- **Domain Mapping**: Fixed domain name abbreviations for proper subject lookup

## 📋 **Advanced Features**

### **📊 Fixed Question Distribution System**
- **Consistent Structure**: All papers use the same question distribution
- **Exact 40 Marks**: Always sums to exactly 40 marks total
- **Fixed Question Counts**:
  - 5 MCQ questions (5 × 2 marks = 10 marks)
  - 5 True/False questions (5 × 2 marks = 10 marks)
  - 2 Long-form questions (2 × 10 marks = 20 marks)
  - Total: 40 marks per paper

### **🧠 Intelligent Dataset Selection**
- **MMLU**: Basic and college-level questions for K-12 and undergraduate
- **MMLU-Pro**: Advanced graduate-level questions with research focus
- **GSM8K**: Mathematics word problems for all levels
- **MBPP+**: Programming and coding questions for CS domains
- **AI2-ARC**: Grade-school science questions for K-12 Science domain
- **PubMedQA**: Medical/health questions for biology and health sciences
- **SQuAD**: Reading comprehension for history, geography, and social sciences

### **📊 Hierarchical Academic Progression**
- **K-12 Level**: Basic concepts using MMLU elementary subjects + AI2-ARC for Science
- **Undergraduate Level**: College-level subjects with MMLU + some MMLU-Pro
- **Graduate Level**: Advanced research questions using MMLU-Pro + specialized datasets

### **🔄 Smart Question Generation**
- **MCQ Questions**: Multiple choice with 4 options (A, B, C, D)
- **True/False Questions**: Generated from MCQ answers with logical statements (LLM-enhanced)
- **Long-form Questions**: Problem-solving and essay questions (LLM-converted from MCQ)
- **Adaptive Filtering**: Smart question selection with fallback mechanisms
- **Smart Routing**: Automatic function selection based on dataset type (MMLU, MMLU-Pro, AI2-ARC)
- **Domain-Specific Filtering**: Questions filtered by domain subjects for relevance

### **🛡️ Robust Error Handling**
- **Comprehensive Validation**: Early detection of empty datasets and configuration issues
- **Graceful Fallbacks**: Automatic dataset switching when questions run out
- **Detailed Logging**: Complete progress tracking and error reporting in `logs/integrity_shield.log`
- **Realistic Domain Mapping**: Prevents unrealistic academic level combinations
- **Fixed Distribution Validation**: Ensures all papers have exactly 5 MCQ, 5 TF, 2 Long-form (40 marks total)
- **Empty Pool Detection**: Warns when datasets are filtered to empty lists with actionable guidance

### **📈 Scalability**
- **Configurable Paper Count**: Adjust `papers_per_domain_level` in `config.yaml` (default: 5)
- **Question Cycling**: When pools are exhausted, they reshuffle and cycle for variety
- **Dataset Caching**: Datasets loaded once and reused for all papers
- **LLM Caching**: Conversion results cached to reduce API costs
- **Memory Efficient**: Efficient dataset loading with minimal memory footprint

### **🤖 LLM-Based Question Conversion**
- **Long-Form Questions**: Uses GPT-4o-mini to convert MCQ questions into well-framed long-form questions
- **True/False Questions**: Uses GPT-4o-mini to intelligently frame True/False statements from MCQ questions
- **Domain-Aware**: Questions are adapted to the specific domain and academic level
- **Caching**: Conversion results are cached to reduce API costs and improve performance
- **Error Handling**: Graceful fallback to rule-based conversion if API fails
- **Configuration**: LLM settings can be customized in `config.yaml`
- **Cost-Effective**: Caching ensures each unique question is only converted once

#### **LLM Setup Instructions**
1. **Get OpenAI API Key**: Sign up at [OpenAI](https://platform.openai.com/) and get your API key
2. **Create `.env` file**: Create a `.env` file in the project root directory
3. **Add API Key**: Add the following line to `.env`:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. **Configure LLM Settings** (optional): Edit `config.yaml` under the `llm` section:
   ```yaml
   llm:
     provider: "openai"
     model: "gpt-4o-mini"  # or "gpt-3.5-turbo", "gpt-4", etc.
     temperature: 0.7
     max_tokens: 200
     enable_caching: true
     cache_file: "data/cache/llm_conversions.json"
     max_retries: 3
     timeout: 30
   ```

#### **LLM Features**
- **Smart Question Framing**: 
  - **Long-Form**: Converts MCQ questions to appropriate long-form prompts based on question type and domain
  - **True/False**: Generates well-framed true and false statements that make sense independently, rather than just quoting the MCQ answer
- **Academic Level Adaptation**: Questions are tailored to K-12, Undergraduate, or Graduate levels
- **Caching System**: All conversions (both long-form and True/False) are cached in `data/cache/llm_conversions.json` to avoid redundant API calls
- **Retry Logic**: Automatic retry with exponential backoff for rate limits and timeouts
- **Fallback Mechanism**: If LLM conversion fails, system falls back to rule-based conversion

#### **Cost Considerations**
- **GPT-4o-mini**: Very cost-effective (~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens)
- **Caching**: Significantly reduces costs by avoiding duplicate conversions
- **Estimated Cost**: 
  - ~$0.001-0.002 per long-form question (with caching, much less for repeated questions)
  - ~$0.001-0.002 per True/False question pair (with caching, much less for repeated questions)

## ⚙️ **Configuration**

The `config.yaml` file contains:
- **Dataset Sources**: URLs and configurations for MMLU, MMLU-Pro, GSM8K, MBPP+, AI2-ARC, PubMedQA, SQuAD
- **Domain Academic Levels**: Realistic academic level mappings for each domain
- **Hierarchical Subject Mappings**: Subject-to-dataset mappings for each domain/level
- **Fixed Question Distribution**: Consistent 5 MCQ, 5 TF, 2 Long-form across all papers
- **Generation Settings**: Number of papers per domain-level combination (default: 5)
- **Fallback Mechanisms**: Automatic dataset switching and error recovery
- **LLM Configuration**: OpenAI API settings and caching options
- **Logging Configuration**: Log level, file path, rotation settings

### **Key Configuration Options**

**Papers per Domain-Level:**
```yaml
domain_generation:
  papers_per_domain_level: 25  # Change this to generate more/fewer papers (default: 25)
```

**Domains to Generate:**
```yaml
domain_generation:
  domains_to_generate: [
    "science", "mathematics", "physics", "chemistry", "biology",
    "computer_science", "engineering", "economics", "psychology", 
    "philosophy", "health_sciences", "business_administration", 
    "history", "geography", "sociology", "political_science", 
    "religious_studies", "machine_learning", "cybersecurity", 
    "astronomy", "constitutional_law"
  ]
```

## 📈 **Output Structure**

Each generated paper includes:
- **LaTeX Source** (`*.tex`) in `latex_documents/`
- **PDF Document** (`*.pdf`) in `pdf_documents/`
- **Answer Keys** (`*_answer_key.pdf` + LaTeX) in `answer_keys/` for quick grading
- **Gold Labels** (`*_gold.json`) with correct answers and metadata
- **Metadata** (`*_metadata.json`) with paper information and question sources

**Output Organization:**
```
output/
├── <domain>/
│   ├── <level>/
│   │   ├── latex_documents/
│   │   ├── pdf_documents/
│   │   ├── answer_keys/
│   │   │   ├── latex/
│   │   │   └── pdf/
│   │   └── JSON_output/
```

## 🔧 **Technical Details**

### **System Requirements**
- **Python 3.8+** required
- **LaTeX** installation required for PDF generation
- **Hugging Face Datasets** for dataset integration
- **Memory Efficient** dataset loading and processing

### **Architecture**
- **Hierarchical Generation**: Domain → Academic Level → Fixed Question Distribution
- **Parallel Processing**: ThreadPoolExecutor-based parallelization at (domain, level) granularity
  - Each worker generates all papers for one (domain, level) combination
  - Thread-safe random number generation with unique seeds per task
  - Semaphore-based PDF compilation limiting
- **Dataset Pool Management**: Intelligent question selection and cycling across 7 datasets
- **Question Type Routing**: Automatic function selection based on dataset type
- **Fixed Distribution System**: All papers use consistent 5 MCQ, 5 TF, 2 Long-form distribution (40 marks)
- **Error Recovery**: Graceful fallbacks and comprehensive error handling
- **Modular Design**: Separate functions for different dataset types and question formats
- **Comprehensive Logging**: Detailed logging at every step for easy debugging
- **Thread-Safety**: Local `random.Random` instances prevent race conditions in parallel execution

### **Dataset Pool Features**
- **Automatic Cycling**: When a pool is exhausted, it reshuffles and cycles for variety
- **Domain Filtering**: Questions filtered by domain subjects for relevance
- **Empty Pool Detection**: Early warnings when datasets are filtered to empty lists
- **Fallback Mechanisms**: Automatic switching to alternative datasets when primary source is unavailable

## 📞 **Support & Troubleshooting**

### **Common Issues**

#### **"No subjects found for domain X level Y"**
- **Cause**: Domain-level combination doesn't have subjects configured in `config.yaml`
- **Solution**: Check `config.yaml` for hierarchical mappings (e.g., `mmlu_<domain>_<level>`)

#### **"MMLU-Pro dataset is empty"**
- **Cause**: MMLU-Pro dataset filtered to empty list for that domain-level combination
- **Solution**: System automatically falls back to regular MMLU. Check logs for warnings.

#### **"MMLU dataset is empty"**
- **Cause**: No MMLU items match the configured subjects for that domain-level
- **Solution**: Check if subjects in `config.yaml` match actual subject names in the dataset

#### **"Ran out of questions"**
- **Cause**: Dataset pool exhausted (shouldn't happen with cycling, but may indicate small filtered pool)
- **Solution**: System automatically cycles and reshuffles. Check if filtered pool is too small.

#### **LaTeX compilation errors**
- **Cause**: LaTeX installation issues or file permissions
- **Solution**: Check LaTeX installation and file permissions. Logs saved next to LaTeX files.

#### **"list index out of range"**
- **Cause**: Empty options list accessed (should be fixed with recent updates)
- **Solution**: Check logs for details. System now validates empty lists before access.

#### **System crash/restart during parallelization**
- **Cause**: Too many concurrent `pdflatex` processes causing resource exhaustion
- **Solution**: Reduce `--max-compile-jobs` to 2-4 for laptops, or reduce `--workers` count
- **Prevention**: Use conservative settings: `--workers 4 --max-compile-jobs 2` for laptops

#### **Thread-safety errors or corrupted random states**
- **Cause**: Using global `random` module in parallel execution (should be fixed)
- **Solution**: Ensure you're using the latest version with thread-safe `random.Random` instances
- **Note**: Fixed in v6.0 - each `DatasetPool` uses its own `random.Random` instance

### **Logging**
- **Detailed logs**: Check `logs/integrity_shield.log` for comprehensive error information
- **Progress tracking**: Real-time generation status and success rates
- **Error recovery**: Automatic fallback and retry mechanisms logged
- **Verbose mode**: Use `--verbose` flag for DEBUG level logging

### **Performance Metrics**
- **Success Rate**: High (1074+ papers generated successfully in latest run)
- **Generation Time**: 
  - Sequential: ~20-30 minutes for full pipeline run (all domains, 25 papers each)
  - Parallel (6 workers, 4 compile jobs): ~5-10 minutes for same workload (3-5x speedup)
- **Memory Usage**: Efficient dataset loading with minimal memory footprint
- **Fixed Distribution**: 100% consistency - all papers have exactly 5 MCQ, 5 TF, 2 Long-form (40 marks)
- **Dataset Coverage**: 7 datasets (MMLU, MMLU-Pro, GSM8K, MBPP+, AI2-ARC, PubMedQA, SQuAD)
- **Scalability**: Can generate thousands of papers by adjusting `papers_per_domain_level`
- **Parallelization**: Thread-safe parallel execution with configurable worker counts
- **Resource Management**: Semaphore-based PDF compilation limiting prevents system overload

### **Recent Performance (Latest Run)**
- **Total Papers Generated**: 1074+ papers
- **Failed Papers**: 1 (mathematics Undergraduate - specific paper failure)
- **Parallelization**: 6 workers, 4 concurrent PDF compilations
- **All Domains Working**: All 22 domains generating successfully
- **Folder Structure**: Hierarchical organization by domain and academic level

## ⚡ **Parallelization & Performance**

### **Parallel Processing Architecture**
The pipeline uses `ThreadPoolExecutor` to parallelize paper generation at the (domain, level) granularity:

- **Worker Granularity**: Each worker generates all papers for one (domain, level) combination
- **Benefits**: 
  - Reuses `DatasetPool` per worker (reduces dataset loading overhead)
  - Natural load balancing across domains
  - Lower overhead compared to per-paper parallelization
- **Thread-Safety**: 
  - Local `random.Random` instances per `DatasetPool` (not global `random`)
  - Unique seeds per (domain, level) combination for deterministic results
  - Prevents race conditions and corrupted random states

### **PDF Compilation Limiting**
- **Semaphore-Based**: Uses `threading.Semaphore` to limit concurrent `pdflatex` processes
- **Purpose**: Prevents resource exhaustion and system crashes
- **Default**: 4 concurrent PDF compilations (configurable via `--max-compile-jobs`)
- **Recommendation**: 
  - Laptops: 2-4 concurrent compilations
  - Desktops: 4-6 concurrent compilations
  - Servers: 6-8 concurrent compilations

### **Performance Tuning**
```bash
# Conservative (laptops, low-end machines)
python main.py --skip-download --workers 4 --max-compile-jobs 2

# Balanced (default, most machines)
python main.py --skip-download --workers 6 --max-compile-jobs 4

# Aggressive (high-end desktops/servers)
python main.py --skip-download --workers 8 --max-compile-jobs 6
```

### **Expected Performance**
- **Sequential**: ~20-30 minutes for 1,125 papers (45 domain-level combinations × 25 papers)
- **Parallel (6 workers, 4 compile jobs)**: ~5-10 minutes for same workload
- **Speedup**: 3-5x faster with parallelization
- **Resource Usage**: 
  - CPU: Moderate (6 workers + 4 pdflatex processes)
  - Memory: Efficient (datasets loaded once per worker)
  - Disk I/O: Moderate (concurrent file writes)

---

**Last Updated**: December 22, 2025  
**Version**: Enhanced Pipeline v6.0  
**Status**: ✅ Production Ready - High Success Rate with Parallel Processing & Comprehensive Error Handling
