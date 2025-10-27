# IntegrityShield Enhanced Pipeline

## 🚀 **Advanced Hierarchical Question Paper Generation System**

This pipeline generates realistic academic question papers across 19 domains with intelligent academic level differentiation using advanced dataset selection from MMLU, MMLU-Pro, GSM8K, and MBPP+ datasets.

## 📊 **Current Performance**
- **✅ 97.8% Success Rate** (45/46 papers generated successfully)
- **📚 19 Domains** supported with realistic academic level mappings
- **🎓 Smart Academic Levels**: K-12, Undergraduate, Graduate (domain-appropriate)
- **📄 40 Marks** per paper (MCQ, True/False, Long-form questions)
- **🔄 Full Dataset Integration**: MMLU, MMLU-Pro, GSM8K, MBPP+ working seamlessly

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

## 🎯 **Supported Domains with Realistic Academic Levels**

### **📚 Full Academic Spectrum (K-12, Undergraduate, Graduate)**
1. **Mathematics** - Complete progression from basic to advanced
2. **Physics** - K-12 fundamentals to graduate research
3. **Chemistry** - Elementary concepts to advanced chemistry
4. **Biology** - Basic biology to graduate-level studies
5. **Computer Science** - Programming basics to advanced CS
6. **Economics** - Economic concepts to graduate economics
7. **Health** - Health education to advanced health studies
8. **History** - Historical knowledge to graduate research
9. **Geography** - Geographic concepts to advanced geography

### **🎓 Undergraduate & Graduate Only (Realistic Advanced Domains)**
10. **Engineering** - Undergraduate to graduate engineering
11. **Psychology** - College-level to graduate psychology
12. **Philosophy** - Undergraduate to graduate philosophy
13. **Business** - Business studies to graduate business
14. **Sociology** - Undergraduate to graduate sociology
15. **Political Science** - Political studies to graduate level
16. **Religious Studies** - Undergraduate to graduate studies
17. **Machine Learning** - Advanced undergraduate to graduate ML
18. **Cybersecurity** - Advanced undergraduate to graduate cybersecurity

### **🔬 Graduate Only (Highly Specialized)**
19. **Astronomy** - Graduate-level astronomy only

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

## 📋 **Advanced Features**

### **🧠 Intelligent Dataset Selection**
- **MMLU**: Basic and college-level questions for K-12 and undergraduate
- **MMLU-Pro**: Advanced graduate-level questions with research focus
- **GSM8K**: Mathematics word problems for all levels
- **MBPP+**: Programming and coding questions for CS domains

### **📊 Hierarchical Academic Progression**
- **K-12 Level**: Basic concepts using MMLU elementary subjects
- **Undergraduate Level**: College-level subjects with MMLU + some MMLU-Pro
- **Graduate Level**: Advanced research questions using MMLU-Pro + specialized datasets

### **🔄 Smart Question Generation**
- **MCQ Questions**: Multiple choice with 4 options (A, B, C, D)
- **True/False Questions**: Generated from MCQ answers with logical statements
- **Long-form Questions**: Problem-solving and essay questions
- **Adaptive Filtering**: Smart question selection with fallback mechanisms

### **🛡️ Robust Error Handling**
- **97.8% Success Rate**: Comprehensive error handling and recovery
- **Graceful Fallbacks**: Automatic dataset switching when questions run out
- **Detailed Logging**: Complete progress tracking and error reporting
- **Realistic Domain Mapping**: Prevents unrealistic academic level combinations

## ⚙️ **Configuration**

The `config.yaml` file contains:
- **Dataset Sources**: URLs and configurations for MMLU, MMLU-Pro, GSM8K, MBPP+
- **Domain Academic Levels**: Realistic academic level mappings for each domain
- **Hierarchical Subject Mappings**: Subject-to-dataset mappings for each domain/level
- **Question Combinations**: MCQ, True/False, Long-form question distributions
- **Generation Settings**: Number of papers per domain-level combination
- **Fallback Mechanisms**: Automatic dataset switching and error recovery

## 📈 **Output Structure**

Each generated paper includes:
- **LaTeX Source** (`*.tex`) in `latex_documents/`
- **PDF Document** (`*.pdf`) in `pdf_documents/`
- **Gold Labels** (`*_gold.json`) with correct answers and metadata

## 🔧 **Technical Details**

### **System Requirements**
- **Python 3.8+** required
- **LaTeX** installation required for PDF generation
- **Hugging Face Datasets** for MMLU-Pro integration
- **Memory Efficient** dataset loading and processing

### **Architecture**
- **Hierarchical Generation**: Domain → Academic Level → Question Type routing
- **Dataset Pool Management**: Intelligent question selection and cycling
- **Question Type Routing**: Automatic function selection based on dataset type
- **Error Recovery**: Graceful fallbacks and comprehensive error handling
- **Modular Design**: Separate functions for MMLU and MMLU-Pro question generation

## 📞 **Support & Troubleshooting**

### **Common Issues**
- **"No combination found"**: Expected for unrealistic domain-level combinations (e.g., Economics K-12)
- **"Ran out of questions"**: Automatic fallback mechanisms handle this gracefully
- **LaTeX compilation errors**: Check LaTeX installation and file permissions

### **Logging**
- **Detailed logs**: Check `logs/integrity_shield.log` for comprehensive error information
- **Progress tracking**: Real-time generation status and success rates
- **Error recovery**: Automatic fallback and retry mechanisms

### **Performance Metrics**
- **Success Rate**: 97.8% (45/46 papers generated successfully)
- **Generation Time**: ~3-4 minutes for full pipeline run
- **Memory Usage**: Efficient dataset loading with minimal memory footprint

---

**Last Updated**: October 27, 2025  
**Version**: Advanced Hierarchical Pipeline v3.0  
**Status**: ✅ Production Ready - All Major Issues Resolved