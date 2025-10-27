# IntegrityShield Enhanced Pipeline

## 🚀 **Enhanced Question Paper Generation System**

This pipeline generates realistic academic question papers across 19 domains and 3 academic levels using intelligent dataset selection from MMLU, MMLU-Pro, GSM8K, and MBPP+ datasets.

## 📊 **Current Performance**
- **✅ 96.5% Success Rate** (55/57 papers generated successfully)
- **📚 19 Domains** supported
- **🎓 3 Academic Levels**: K-12, Undergraduate, Graduate
- **📄 40 Marks** per paper (MCQ, True/False, Long-form questions)

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
│   │   ├── gsm8k_mcq.json
│   │   └── mbpp_plus.json
│   └── gsm_mcq/             # GSM8K MCQ data
│
├── src/                     # Source code
│   ├── data_processing/     # Dataset loading and processing
│   ├── pdf_generation/      # LaTeX and PDF generation
│   └── utils/               # Configuration and logging utilities
│
├── output/                  # Generated papers (hierarchical structure)
│   ├── mathematics/
│   │   ├── k12/
│   │   │   ├── latex_documents/
│   │   │   └── pdf_documents/
│   │   ├── undergraduate/
│   │   └── graduate/
│   ├── physics/
│   ├── chemistry/
│   └── ... (all 19 domains)
│
├── docs/                    # Documentation
├── examples/                # Example files
└── logs/                    # Log files
```

## 🎯 **Supported Domains**

1. **Mathematics** - K-12, Undergraduate, Graduate
2. **Physics** - K-12, Undergraduate, Graduate  
3. **Chemistry** - K-12, Undergraduate, Graduate
4. **Biology** - K-12, Undergraduate, Graduate
5. **Computer Science** - K-12, Undergraduate, Graduate
6. **Engineering** - K-12, Undergraduate, Graduate
7. **Economics** - K-12, Undergraduate, Graduate
8. **Psychology** - K-12, Undergraduate, Graduate
9. **Philosophy** - K-12, Undergraduate, Graduate
10. **Health** - K-12, Undergraduate, Graduate
11. **Business** - K-12, Undergraduate, Graduate
12. **History** - K-12, Undergraduate, Graduate
13. **Geography** - K-12, Undergraduate, Graduate
14. **Sociology** - K-12, Undergraduate, Graduate
15. **Political Science** - K-12, Undergraduate, Graduate
16. **Religious Studies** - K-12, Undergraduate, Graduate
17. **Machine Learning** - K-12, Undergraduate, Graduate
18. **Cybersecurity** - K-12, Undergraduate, Graduate
19. **Astronomy** - Graduate only (realistic academic level)

## 🚀 **Quick Start**

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download datasets (first time only):**
   ```bash
   python main.py --count 1
   ```

3. **Generate papers:**
```bash
   python main.py --count 1 --skip-download --verbose
   ```

## 📋 **Features**

- **🎯 Intelligent Dataset Selection**: Automatically selects appropriate datasets (MMLU, MMLU-Pro, GSM8K, MBPP+) based on subject and academic level
- **📊 Hierarchical Generation**: Generates papers by domain and academic level with realistic difficulty progression
- **🔄 Adaptive Filtering**: Smart question filtering with fallback mechanisms for maximum success rate
- **📁 Organized Output**: Clean directory structure with domain/academic_level organization
- **🛡️ Error Handling**: Comprehensive error handling with detailed progress tracking
- **📝 LaTeX Generation**: High-quality PDF generation with proper academic formatting

## ⚙️ **Configuration**

The `config.yaml` file contains:
- **Dataset Sources**: URLs and configurations for all datasets
- **Hierarchical Mappings**: Subject-to-dataset mappings for each domain/level
- **Question Combinations**: MCQ, True/False, Long-form question distributions
- **Generation Settings**: Number of papers per domain-level combination

## 📈 **Output Structure**

Each generated paper includes:
- **LaTeX Source** (`*.tex`) in `latex_documents/`
- **PDF Document** (`*.pdf`) in `pdf_documents/`
- **Gold Labels** (`*_gold.json`) with correct answers and metadata

## 🔧 **Technical Details**

- **Python 3.8+** required
- **LaTeX** installation required for PDF generation
- **Hugging Face Datasets** for MMLU-Pro integration
- **Robust Error Handling** with graceful fallbacks
- **Memory Efficient** dataset loading and processing

## 📞 **Support**

For issues or questions, check the logs in `logs/integrity_shield.log` for detailed error information.

---

**Last Updated**: October 27, 2025  
**Version**: Enhanced Hierarchical Pipeline v2.0