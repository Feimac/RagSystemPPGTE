import re
import logging
import os
import fitz  # PyMuPDF
import pdfplumber
from unidecode import unidecode
import chardet
from collections import Counter
import hashlib
import tempfile
from docling.document_converter import DocumentConverter

# Configuração avançada de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("universal_pdf_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UniversalPDFProcessor:
    def __init__(self):
        self.error_patterns = Counter()
        self.correction_map = {}
        self.detected_encoding = None
        self.language = "auto"
        self.sanitization_attempted = False
        
    def detect_language(self, text: str) -> str:
        """Detecta o idioma principal do texto"""
        pt_indicators = sum(text.lower().count(word) for word in ['o', 'a', 'de', 'que', 'e'])
        en_indicators = sum(text.lower().count(word) for word in ['the', 'and', 'of', 'to', 'in'])
        if pt_indicators > en_indicators:
            return "portuguese"
        elif en_indicators > pt_indicators:
            return "english"
        return "unknown"

    def extract_text(self, file_path: str) -> str:
        methods = [
            self._extract_with_fitz,
            self._extract_with_pdfplumber
        ]
        best_text = ""
        best_score = 0
        for method in methods:
            try:
                text = method(file_path)
                if not text:
                    continue
                score = self._evaluate_text_quality(text)
                if score > best_score:
                    best_text = text
                    best_score = score
            except Exception as e:
                logger.warning(f"{method.__name__} failed: {str(e)}")
        if not best_text:
            logger.error("Todas as técnicas de extração falharam")
            return ""
        self.language = self.detect_language(best_text)
        logger.info(f"Idioma detectado: {self.language}")
        return best_text

    def _extract_with_fitz(self, file_path: str) -> str:
        text = ""
        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text("text", sort=True) + "\n"
            return text
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {str(e)}")
            return ""

    def _extract_with_pdfplumber(self, file_path: str) -> str:
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {str(e)}")
            return ""

    def _evaluate_text_quality(self, text: str) -> float:
        if len(text) < 100:
            return 0
        valid_chars = sum(c.isalnum() or c.isspace() or c in ',.;:!?()[]{}@#$%&*_+-=/' for c in text)
        valid_ratio = valid_chars / len(text)
        space_ratio = text.count(' ') / len(text)
        problem_chars = sum(text.count(char) for char in ['�', '\x00', '\ufffd'])
        problem_ratio = problem_chars / len(text)
        score = (valid_ratio * 0.6 + 
                 (0.1 if 0.1 < space_ratio < 0.3 else 0) * 0.2 + 
                 (1 - min(problem_ratio * 10, 1)) * 0.2)
        return score

    def analyze_errors(self, text: str):
        encoding_info = chardet.detect(text.encode('utf-8', errors='replace'))
        self.detected_encoding = encoding_info['encoding'] or 'utf-8'
        logger.info(f"Encoding detectado: {self.detected_encoding} (confiança: {encoding_info['confidence']:.2f})")
        invalid_chars = re.findall(r'[^\w\s.,;:?!@#$%&*()_+=\-[\]{}\\|"\'<>/À-ÿ]', text)
        self.error_patterns.update(invalid_chars)
        if self.language == "portuguese":
            pt_errors = re.findall(r'Ã[çãáéóõêâàúí]|PîS|Gradua,ÌO|CAPêTULO', text)
            self.error_patterns.update(pt_errors)
        elif self.language == "english":
            en_errors = re.findall(r'â€|â€“|â€™|â€œ|â€', text)
            self.error_patterns.update(en_errors)
        hyphenated = re.findall(r'\b\w+-\s*\n\s*\w+\b', text)
        self.error_patterns.update(["hyphenated_word"] * len(hyphenated))
        common_fixes = {
            'Ã§': 'ç', 'Ã£': 'ã', 'Ã¡': 'á', 'Ã©': 'é', 'Ã³': 'ó',
            'Ãµ': 'õ', 'Ãª': 'ê', 'Ã¢': 'â', 'Ã ': 'à', 'Ãº': 'ú',
            'Ã­': 'í', 'PîS': 'Pós', 'Gradua,ÌO': 'Graduação',
            'CAPêTULO': 'CAPÍTULO',
            'â€œ': '"', 'â€': '"', 'â€™': "'", 'â€“': '-'
        }
        for error in self.error_patterns:
            if error in common_fixes:
                self.correction_map[error] = common_fixes[error]
            else:
                corrected = unidecode(error)
                if corrected != error and len(corrected) > 0:
                    self.correction_map[error] = corrected

    def auto_correct(self, text: str) -> str:
        if not text.isascii():
            text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
        for error, correction in self.correction_map.items():
            text = text.replace(error, correction)
        text = re.sub(r'(\b\w+)-\s*\n\s*(\w+\b)', r'\1\2', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(\w)\s*-\s*(\w)', r'\1-\2', text)
        if self.language == "portuguese":
            text = re.sub(r'([a-zà-ÿ])\.\s([A-ZÀ-Ÿ])', r'\1. \n\n\2', text)
            text = re.sub(r'Art\.\s*(\d+)[º°]?', r'**Art. \1**', text)
        return text

    def enhance_structure(self, text: str) -> str:
        text = re.sub(r'\n(\s*)(CAP[ÍI]TULO|CHAPTER)\s+([IVXLCDM]+)(\s*)', r'\n## \2 \3\n', text)
        text = re.sub(r'\n(\d+\.\d+\.?\s+)([^\n]+)', r'\n### \1\2\n', text)
        text = re.sub(r'\n(\s*[-*•])\s+', r'\n- ', text)
        return text

    def sanitize_text(self, text: str) -> str:
        sanitized = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        sanitized = sanitized.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
        replacements = {
            '\ufffd': '',
            '\xad': '',
            '\u200b': '',
        }
        for char, replacement in replacements.items():
            sanitized = sanitized.replace(char, replacement)
        sanitized = re.sub(r'\s+', ' ', sanitized)
        sanitized = re.sub(r'\n{3,}', '\n\n', sanitized)
        return sanitized

    def fallback_sanitization(self, text: str) -> str:
        try:
            encoding_info = chardet.detect(text.encode('latin1', errors='replace'))
            detected_encoding = encoding_info['encoding'] or 'latin1'
            recoded = text.encode(detected_encoding, errors='ignore').decode('utf-8', errors='ignore')
            return self.sanitize_text(recoded)
        except:
            return text.encode('ascii', errors='ignore').decode('ascii', 'ignore')

    def process_with_docling(self, text: str) -> str:
        """Processa o texto com Docling para formatação final com tratamento robusto de encoding"""
        try:
            sanitized_text = self.sanitize_text(text)
            # Gerar temporário com extensão suportada (.md)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp:
                tmp.write(sanitized_text)
                tmp_path = tmp.name
            converter = DocumentConverter()
            result = converter.convert(tmp_path)
            md_text = result.document.export_to_markdown()
            os.unlink(tmp_path)
            return md_text
        except UnicodeEncodeError as uee:
            logger.error(f"Erro de codificação: {str(uee)}")
            return self.fallback_sanitization(text)
        except Exception as e:
            logger.error(f"Falha no processamento Docling: {str(e)}")
            return self.fallback_sanitization(text)

    def process_file(self, input_path: str, output_dir: str):
        logger.info(f"Iniciando processamento: {os.path.basename(input_path)}")
        self.sanitization_attempted = False
        if not os.path.exists(input_path):
            logger.error(f"Arquivo não encontrado: {input_path}")
            return None
        raw_text = self.extract_text(input_path)
        if not raw_text:
            logger.error("Falha na extração de texto")
            return None
        logger.info(f"Texto extraído ({len(raw_text)} caracteres)")
        self.analyze_errors(raw_text)
        logger.info(f"Padrões de erro detectados: {self.error_patterns.most_common(5)}")
        corrected_text = self.auto_correct(raw_text)
        structured_text = self.enhance_structure(corrected_text)
        final_text = self.process_with_docling(structured_text)
        if "�" in final_text and not self.sanitization_attempted:
            logger.warning("Texto contém caracteres inválidos - aplicando sanitização profunda")
            self.sanitization_attempted = True
            sanitized_text = self.fallback_sanitization(raw_text)
            final_text = self.process_with_docling(sanitized_text)
        output_path = os.path.join(output_dir, os.path.basename(input_path) + ".md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_text)
        logger.info(f"Documento processado salvo em: {output_path}")
        text_hash = hashlib.md5(final_text.encode()).hexdigest()
        logger.info(f"MD5 do conteúdo: {text_hash}")
        return final_text

def main():
    PDF_FILES = ['Regulamento_Aprovado_2019.pdf']
    OUTPUT_DIR = "processed_documents"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    processor = UniversalPDFProcessor()
    for pdf_file in PDF_FILES:
        result = processor.process_file(pdf_file, OUTPUT_DIR)
        processor = UniversalPDFProcessor()
        if result:
            logger.info(f"Processamento concluído com sucesso para {pdf_file}")
        else:
            logger.error(f"Falha no processamento de {pdf_file}")

if __name__ == "__main__":
    main()
