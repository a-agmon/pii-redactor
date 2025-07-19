"""
Dataset Creation Module for PII Redaction Model

This module handles the creation of synthetic multilingual datasets for training
a PII redaction model. It includes specialized support for Hebrew language with
proper handling of RTL text and Hebrew-specific PII patterns.

Classes:
    HebrewPIIGenerator: Generates synthetic Hebrew PII data
    MultilingualPIIDataset: Creates multilingual datasets with BIO tagging
    PIIDataProcessor: Processes and tokenizes data for model training
"""

import random
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import yaml
import logging
from faker import Faker
from datasets import Dataset, DatasetDict
import pandas as pd
from tqdm import tqdm
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PIIEntity:
    """Data class for PII entities"""
    entity_type: str
    value: str
    start: int
    end: int


class HebrewPIIGenerator:
    """
    Generate synthetic Hebrew PII data with realistic patterns.
    
    This class creates Hebrew-specific PII entities including names,
    ID numbers, phone numbers, addresses, and other personal information
    following Israeli standards and formats.
    """
    
    def __init__(self):
        """Initialize Hebrew PII generator with predefined data"""
        # Common Hebrew first names
        self.hebrew_first_names = [
            "אלון", "שרה", "דוד", "רחל", "משה", "לאה", "יוסף", "מרים",
            "אברהם", "רבקה", "יצחק", "אסתר", "יעקב", "חנה", "נח", "דינה",
            "בנימין", "תמר", "שמואל", "נעמי", "אליהו", "רות", "דניאל", "עדינה",
            "גבריאל", "שושנה", "מיכאל", "יעל", "אורי", "טליה", "עומר", "נויה"
        ]
        
        # Common Hebrew surnames
        self.hebrew_surnames = [
            "כהן", "לוי", "מזרחי", "פרץ", "ביטון", "אזולאי", "דהן", "אברהם",
            "חדד", "גבאי", "אוחיון", "בן דוד", "מלכה", "אשכנזי", "ישראלי", "ברוך",
            "סעדון", "חיים", "נחום", "שלום", "בן שמעון", "זכריה", "אליאס", "יוסף"
        ]
        
        # Israeli city names
        self.israeli_cities = [
            "תל אביב", "ירושלים", "חיפה", "ראשון לציון", "פתח תקווה", "אשדוד",
            "נתניה", "באר שבע", "בני ברק", "חולון", "רמת גן", "אשקלון",
            "רחובות", "בת ים", "כפר סבא", "הרצליה", "מודיעין", "רעננה"
        ]
        
        # Street names
        self.street_names = [
            "הרצל", "ויצמן", "רוטשילד", "בן גוריון", "ז'בוטינסקי", "אלנבי",
            "דיזנגוף", "בן יהודה", "המלך ג'ורג'", "שדרות ירושלים", "הארבעה",
            "סוקולוב", "ביאליק", "אחד העם", "הנביאים", "יפו"
        ]
        
        # Email domains
        self.email_domains = [
            "gmail.com", "walla.co.il", "hotmail.com", "yahoo.com",
            "outlook.com", "mail.huji.ac.il", "technion.ac.il", "tau.ac.il"
        ]
        
        # Phone prefixes for Israeli mobile numbers
        self.phone_prefixes = ["050", "052", "053", "054", "055", "058"]
        
    def generate_israeli_id(self) -> str:
        """Generate a valid-looking Israeli ID number (9 digits)"""
        # Generate 8 random digits
        id_digits = [random.randint(0, 9) for _ in range(8)]
        
        # Calculate check digit using Luhn algorithm
        total = 0
        for i, digit in enumerate(id_digits):
            if i % 2 == 0:
                doubled = digit * 2
                total += doubled if doubled < 10 else doubled - 9
            else:
                total += digit
        
        check_digit = (10 - (total % 10)) % 10
        id_digits.append(check_digit)
        
        return ''.join(map(str, id_digits))
    
    def generate_israeli_phone(self) -> str:
        """Generate Israeli phone number"""
        prefix = random.choice(self.phone_prefixes)
        suffix = ''.join([str(random.randint(0, 9)) for _ in range(7)])
        return f"{prefix}-{suffix[:3]}-{suffix[3:]}"
    
    def generate_hebrew_address(self) -> str:
        """Generate Hebrew address"""
        street = random.choice(self.street_names)
        number = random.randint(1, 150)
        city = random.choice(self.israeli_cities)
        return f"{street} {number}, {city}"
    
    def generate_email(self, name: str) -> str:
        """Generate email address based on name"""
        # Transliterate common Hebrew names to English
        transliteration_map = {
            "אלון": "alon", "שרה": "sarah", "דוד": "david", "רחל": "rachel",
            "משה": "moshe", "לאה": "leah", "יוסף": "yosef", "מרים": "miriam"
        }
        
        # Get transliterated name or use a random string
        eng_name = transliteration_map.get(name.split()[0], f"user{random.randint(100, 999)}")
        domain = random.choice(self.email_domains)
        
        return f"{eng_name}{random.randint(1, 99)}@{domain}"
    
    def generate_credit_card(self) -> str:
        """Generate credit card number pattern"""
        # Generate pattern like ****-****-****-1234
        last_four = ''.join([str(random.randint(0, 9)) for _ in range(4)])
        return f"****-****-****-{last_four}"
    
    def generate_passport(self) -> str:
        """Generate Israeli passport number"""
        letters = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))
        numbers = ''.join([str(random.randint(0, 9)) for _ in range(7)])
        return f"{letters}{numbers}"
    
    def generate_bank_account(self) -> str:
        """Generate Israeli bank account number"""
        bank = random.randint(10, 20)  # Bank code
        branch = random.randint(100, 999)  # Branch code
        account = random.randint(100000, 999999)  # Account number
        return f"{bank}-{branch}-{account}"
    
    def generate_date_of_birth(self) -> str:
        """Generate date of birth in Hebrew format"""
        day = random.randint(1, 28)
        month = random.randint(1, 12)
        year = random.randint(1950, 2005)
        
        hebrew_months = [
            "ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני",
            "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר"
        ]
        
        return f"{day} ב{hebrew_months[month-1]} {year}"
    
    def generate_license_plate(self) -> str:
        """Generate Israeli license plate number"""
        # New format: XXX-XX-XXX or XX-XXX-XX
        if random.choice([True, False]):
            return f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(100, 999)}"
        else:
            return f"{random.randint(10, 99)}-{random.randint(100, 999)}-{random.randint(10, 99)}"
    
    def get_templates(self) -> List[Dict]:
        """
        Get Hebrew sentence templates with PII placeholders and clean sentences.
        
        Returns:
            List of template dictionaries with template string and entity types
        """
        # PII-containing templates (original)
        pii_templates = [
            {
                "template": "שמי {NAME} ומספר תעודת הזהות שלי הוא {ID_NUMBER}",
                "entities": ["NAME", "ID_NUMBER"]
            },
            {
                "template": "אני {NAME}, הטלפון שלי {PHONE} והמייל {EMAIL}",
                "entities": ["NAME", "PHONE", "EMAIL"]
            },
            {
                "template": "הכתובת של {NAME} היא {ADDRESS}",
                "entities": ["NAME", "ADDRESS"]
            },
            {
                "template": "נולדתי ב-{DATE_OF_BIRTH}, השם שלי {NAME}",
                "entities": ["DATE_OF_BIRTH", "NAME"]
            },
            {
                "template": "מספר הדרכון של {NAME} הוא {PASSPORT}",
                "entities": ["NAME", "PASSPORT"]
            },
            {
                "template": "אני {NAME} ומספר כרטיס האשראי שלי הוא {CREDIT_CARD}",
                "entities": ["NAME", "CREDIT_CARD"]
            },
            {
                "template": "חשבון הבנק של {NAME} הוא {BANK_ACCOUNT}",
                "entities": ["NAME", "BANK_ACCOUNT"]
            },
            {
                "template": "מספר הרכב של {NAME} הוא {LICENSE_PLATE}",
                "entities": ["NAME", "LICENSE_PLATE"]
            },
            {
                "template": "שלום, קוראים לי {NAME} ואני גר ב{ADDRESS}. הטלפון שלי הוא {PHONE}",
                "entities": ["NAME", "ADDRESS", "PHONE"]
            },
            {
                "template": "לפרטים נוספים: {NAME}, ת.ז. {ID_NUMBER}, טלפון {PHONE}",
                "entities": ["NAME", "ID_NUMBER", "PHONE"]
            },
            {
                "template": "בתאריך {DATE_OF_BIRTH} נולד {NAME}, תושב {ADDRESS}",
                "entities": ["DATE_OF_BIRTH", "NAME", "ADDRESS"]
            },
            {
                "template": "ניתן ליצור קשר עם {NAME} במייל {EMAIL} או בטלפון {PHONE}",
                "entities": ["NAME", "EMAIL", "PHONE"]
            }
        ]
        
        # Clean templates (no PII) with technical terms and common words
        clean_templates = [
            {
                "template": "אני אוהב לתכנת בפייתון והספרייה המועדפת עלי היא NumPy",
                "entities": []
            },
            {
                "template": "היום למדתי על למידת מכונה ובינה מלאכותית",
                "entities": []
            },
            {
                "template": "האתר שלי נמצא בכתובת ובו אפשר למצוא מידע על פרויקטים",
                "entities": []
            },
            {
                "template": "אני עובד עם מסד נתונים גדול ומשתמש בכלים שונים לניתוח",
                "entities": []
            },
            {
                "template": "הטכנולוגיה שאני הכי אוהב היא React ו-JavaScript",
                "entities": []
            },
            {
                "template": "המחשב שלי רץ על מערכת הפעלה לינוקס ואני משתמש בטרמינל",
                "entities": []
            },
            {
                "template": "השפה המועדפת עלי לפיתוח היא Python וGo",
                "entities": []
            },
            {
                "template": "אני עובד בחברת טכנולוגיה ומפתח אפליקציות ווב",
                "entities": []
            },
            {
                "template": "הפרויקט שלי כולל שרת Node.js ומסד נתונים MongoDB",
                "entities": []
            },
            {
                "template": "אני אוהב לקרוא ספרים על אלגוריתמים ומבני נתונים",
                "entities": []
            }
        ]
        
        # Mixed templates (PII + regular content)
        mixed_templates = [
            {
                "template": "שלום, אני {NAME} ואני עובד כמתכנת Python ב-Google. אפשר ליצור קשר במייל {EMAIL}",
                "entities": ["NAME", "EMAIL"]
            },
            {
                "template": "המפתח {NAME} יצר ספרייה נהדרת בשם TensorFlow, ניתן ליצור קשר בטלפון {PHONE}",
                "entities": ["NAME", "PHONE"]
            },
            {
                "template": "אני {NAME} ואני אוהב לעבוד עם Docker ו-Kubernetes בפרויקטים שלי",
                "entities": ["NAME"]
            },
            {
                "template": "המהנדס {NAME} פיתח API מעולה ב-Flask, הכתובת שלו היא {ADDRESS}",
                "entities": ["NAME", "ADDRESS"]
            },
            {
                "template": "אני משתמש ב-Git ו-GitHub לניהול הקוד, השם שלי {NAME} ואימייל {EMAIL}",
                "entities": ["NAME", "EMAIL"]
            }
        ]
        
        # Combine all templates with appropriate weights
        all_templates = pii_templates + clean_templates * 2 + mixed_templates
        return all_templates
    
    def generate_value(self, entity_type: str, context: Optional[Dict] = None) -> str:
        """
        Generate a value for a specific PII entity type.
        
        Args:
            entity_type: Type of PII entity to generate
            context: Optional context for consistent entity generation
            
        Returns:
            Generated PII value
        """
        generators = {
            "NAME": lambda: f"{random.choice(self.hebrew_first_names)} {random.choice(self.hebrew_surnames)}",
            "ID_NUMBER": self.generate_israeli_id,
            "PHONE": self.generate_israeli_phone,
            "EMAIL": lambda: self.generate_email(context.get("NAME", "user") if context else "user"),
            "ADDRESS": self.generate_hebrew_address,
            "CREDIT_CARD": self.generate_credit_card,
            "DATE_OF_BIRTH": self.generate_date_of_birth,
            "PASSPORT": self.generate_passport,
            "BANK_ACCOUNT": self.generate_bank_account,
            "LICENSE_PLATE": self.generate_license_plate
        }
        
        generator = generators.get(entity_type)
        if generator:
            return generator()
        else:
            logger.warning(f"Unknown entity type: {entity_type}")
            return "[UNKNOWN]"


class MultilingualPIIDataset:
    """
    Create multilingual PII dataset with BIO tags.
    
    This class generates synthetic PII data in multiple languages with proper
    BIO (Begin-Inside-Outside) tagging for token classification tasks.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize dataset creator.
        
        Args:
            config: Configuration dictionary with dataset parameters
        """
        self.config = config
        self.hebrew_generator = HebrewPIIGenerator()
        
        # Initialize Faker instances for different languages
        self.faker_generators = {
            'en': Faker('en_US'),
            'es': Faker('es_ES'),
            'fr': Faker('fr_FR'),
            'de': Faker('de_DE')
        }
        
        # Add seed for reproducibility
        random.seed(config.get('seed', 42))
        Faker.seed(config.get('seed', 42))
    
    def generate_multilingual_templates(self, language: str) -> List[Dict]:
        """
        Generate templates for different languages.
        
        Args:
            language: Language code (en, es, fr, de)
            
        Returns:
            List of template dictionaries
        """
        if language == 'en':
            # PII-containing templates
            pii_templates = [
                {
                    "template": "My name is {NAME} and my ID number is {ID_NUMBER}",
                    "entities": ["NAME", "ID_NUMBER"]
                },
                {
                    "template": "I am {NAME}, my phone is {PHONE} and email is {EMAIL}",
                    "entities": ["NAME", "PHONE", "EMAIL"]
                },
                {
                    "template": "The address of {NAME} is {ADDRESS}",
                    "entities": ["NAME", "ADDRESS"]
                },
                {
                    "template": "Born on {DATE_OF_BIRTH}, my name is {NAME}",
                    "entities": ["DATE_OF_BIRTH", "NAME"]
                },
                {
                    "template": "Contact {NAME} at {EMAIL} or call {PHONE}",
                    "entities": ["NAME", "EMAIL", "PHONE"]
                }
            ]
            
            # Clean templates (no PII)
            clean_templates = [
                {
                    "template": "I love programming in Python and my favorite library is NumPy",
                    "entities": []
                },
                {
                    "template": "Today I learned about machine learning and artificial intelligence",
                    "entities": []
                },
                {
                    "template": "You can visit my website to find information about my projects",
                    "entities": []
                },
                {
                    "template": "I work with large databases and use various tools for analysis",
                    "entities": []
                },
                {
                    "template": "My favorite technology stack is React and JavaScript",
                    "entities": []
                },
                {
                    "template": "My computer runs on Linux and I use the terminal regularly",
                    "entities": []
                },
                {
                    "template": "My preferred programming language is Python and Go",
                    "entities": []
                },
                {
                    "template": "I work at a tech company developing web applications",
                    "entities": []
                },
                {
                    "template": "My project includes a Node.js server and MongoDB database",
                    "entities": []
                },
                {
                    "template": "I enjoy reading books about algorithms and data structures",
                    "entities": []
                }
            ]
            
            # Mixed templates (PII + regular content)
            mixed_templates = [
                {
                    "template": "Hello, I'm {NAME} and I work as a Python developer at Google. You can reach me at {EMAIL}",
                    "entities": ["NAME", "EMAIL"]
                },
                {
                    "template": "The developer {NAME} created an amazing library called TensorFlow, contact at {PHONE}",
                    "entities": ["NAME", "PHONE"]
                },
                {
                    "template": "I'm {NAME} and I love working with Docker and Kubernetes in my projects",
                    "entities": ["NAME"]
                },
                {
                    "template": "Engineer {NAME} developed an excellent API in Flask, address is {ADDRESS}",
                    "entities": ["NAME", "ADDRESS"]
                },
                {
                    "template": "I use Git and GitHub for code management, my name is {NAME} and email {EMAIL}",
                    "entities": ["NAME", "EMAIL"]
                }
            ]
            
            return pii_templates + clean_templates * 2 + mixed_templates
        elif language == 'es':
            pii_templates = [
                {
                    "template": "Mi nombre es {NAME} y mi número de ID es {ID_NUMBER}",
                    "entities": ["NAME", "ID_NUMBER"]
                },
                {
                    "template": "Soy {NAME}, mi teléfono es {PHONE} y mi correo es {EMAIL}",
                    "entities": ["NAME", "PHONE", "EMAIL"]
                },
                {
                    "template": "La dirección de {NAME} es {ADDRESS}",
                    "entities": ["NAME", "ADDRESS"]
                }
            ]
            clean_templates = [
                {
                    "template": "Me encanta programar en Python y mi biblioteca favorita es NumPy",
                    "entities": []
                },
                {
                    "template": "Trabajo con bases de datos grandes y uso varias herramientas para análisis",
                    "entities": []
                },
                {
                    "template": "Mi tecnología favorita es React y JavaScript",
                    "entities": []
                }
            ]
            return pii_templates + clean_templates * 2
        elif language == 'fr':
            pii_templates = [
                {
                    "template": "Je m'appelle {NAME} et mon numéro d'identification est {ID_NUMBER}",
                    "entities": ["NAME", "ID_NUMBER"]
                },
                {
                    "template": "Je suis {NAME}, mon téléphone est {PHONE} et mon email est {EMAIL}",
                    "entities": ["NAME", "PHONE", "EMAIL"]
                },
                {
                    "template": "L'adresse de {NAME} est {ADDRESS}",
                    "entities": ["NAME", "ADDRESS"]
                }
            ]
            clean_templates = [
                {
                    "template": "J'adore programmer en Python et ma bibliothèque préférée est NumPy",
                    "entities": []
                },
                {
                    "template": "Je travaille avec de grandes bases de données et j'utilise divers outils d'analyse",
                    "entities": []
                },
                {
                    "template": "Ma technologie préférée est React et JavaScript",
                    "entities": []
                }
            ]
            return pii_templates + clean_templates * 2
        elif language == 'de':
            pii_templates = [
                {
                    "template": "Mein Name ist {NAME} und meine ID-Nummer ist {ID_NUMBER}",
                    "entities": ["NAME", "ID_NUMBER"]
                },
                {
                    "template": "Ich bin {NAME}, meine Telefonnummer ist {PHONE} und meine E-Mail ist {EMAIL}",
                    "entities": ["NAME", "PHONE", "EMAIL"]
                },
                {
                    "template": "Die Adresse von {NAME} ist {ADDRESS}",
                    "entities": ["NAME", "ADDRESS"]
                }
            ]
            clean_templates = [
                {
                    "template": "Ich programmiere gerne in Python und meine Lieblingsbibliothek ist NumPy",
                    "entities": []
                },
                {
                    "template": "Ich arbeite mit großen Datenbanken und verwende verschiedene Analysetools",
                    "entities": []
                },
                {
                    "template": "Meine bevorzugte Technologie ist React und JavaScript",
                    "entities": []
                }
            ]
            return pii_templates + clean_templates * 2
        else:
            return []
    
    def generate_multilingual_value(self, entity_type: str, language: str, context: Optional[Dict] = None) -> str:
        """
        Generate PII value for non-Hebrew languages.
        
        Args:
            entity_type: Type of PII entity
            language: Language code
            context: Optional context for consistent generation
            
        Returns:
            Generated PII value
        """
        faker = self.faker_generators.get(language)
        if not faker:
            return "[UNKNOWN]"
        
        generators = {
            "NAME": faker.name,
            "ID_NUMBER": lambda: faker.ssn() if language == 'en' else faker.random_number(digits=9, fix_len=True),
            "PHONE": faker.phone_number,
            "EMAIL": faker.email,
            "ADDRESS": faker.address,
            "CREDIT_CARD": faker.credit_card_number,
            "DATE_OF_BIRTH": lambda: faker.date_of_birth().strftime("%d/%m/%Y"),
            "PASSPORT": lambda: faker.bothify(text='??#######').upper(),
            "BANK_ACCOUNT": lambda: faker.iban(),
            "LICENSE_PLATE": faker.license_plate
        }
        
        generator = generators.get(entity_type)
        if generator:
            return str(generator())
        else:
            return "[UNKNOWN]"
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Simple whitespace tokenization.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # For production, use proper tokenizer from transformers
        # This is a simplified version for demonstration
        return text.split()
    
    def generate_sample(self, language: str = 'he') -> Dict:
        """
        Generate a single training sample with BIO tags.
        
        Args:
            language: Language code for sample generation
            
        Returns:
            Dictionary with text, tokens, and labels
        """
        # Select appropriate templates
        if language == 'he':
            templates = self.hebrew_generator.get_templates()
        else:
            templates = self.generate_multilingual_templates(language)
        
        if not templates:
            logger.warning(f"No templates available for language: {language}")
            return None
        
        # Choose a random template
        template_data = random.choice(templates)
        template = template_data["template"]
        entity_types = template_data["entities"]
        
        # Generate values for entities
        context = {}
        entity_values = {}
        
        for entity_type in entity_types:
            if language == 'he':
                value = self.hebrew_generator.generate_value(entity_type, context)
            else:
                value = self.generate_multilingual_value(entity_type, language, context)
            
            entity_values[entity_type] = value
            context[entity_type] = value
        
        # Fill template with values
        text = template
        entities = []
        
        for entity_type, value in entity_values.items():
            placeholder = f"{{{entity_type}}}"
            start_pos = text.find(placeholder)
            if start_pos != -1:
                text = text.replace(placeholder, value, 1)
                # Store entity position (adjusted for replacement)
                entities.append(PIIEntity(
                    entity_type=entity_type,
                    value=value,
                    start=start_pos,
                    end=start_pos + len(value)
                ))
        
        # Tokenize text
        tokens = self.tokenize_text(text)
        
        # Generate BIO labels
        labels = ["O"] * len(tokens)
        
        # Map entities to token positions
        char_to_token = {}
        current_char = 0
        
        for i, token in enumerate(tokens):
            for j in range(len(token)):
                char_to_token[current_char + j] = i
            current_char += len(token) + 1  # +1 for space
        
        # Assign BIO labels
        for entity in entities:
            # Find tokens that overlap with entity
            start_token = char_to_token.get(entity.start)
            end_token = char_to_token.get(entity.end - 1)
            
            if start_token is not None and end_token is not None:
                labels[start_token] = "B-PII"
                for i in range(start_token + 1, end_token + 1):
                    if i < len(labels):
                        labels[i] = "I-PII"
        
        return {
            "text": text,
            "tokens": tokens,
            "labels": labels,
            "language": language
        }
    
    def create_dataset(self, num_samples: int) -> DatasetDict:
        """
        Create complete dataset with train/val/test splits.
        
        Args:
            num_samples: Total number of samples to generate
            
        Returns:
            HuggingFace DatasetDict with train, validation, and test splits
        """
        logger.info(f"Generating {num_samples} samples...")
        
        # Calculate samples per language
        hebrew_samples = int(num_samples * self.config['dataset']['hebrew_ratio'])
        other_samples = num_samples - hebrew_samples
        other_languages = [lang for lang in self.config['dataset']['languages'] if lang != 'he']
        samples_per_other_lang = other_samples // len(other_languages) if other_languages else 0
        
        # Generate samples
        all_samples = []
        
        # Generate Hebrew samples
        logger.info(f"Generating {hebrew_samples} Hebrew samples...")
        for _ in tqdm(range(hebrew_samples), desc="Hebrew samples"):
            sample = self.generate_sample('he')
            if sample:
                all_samples.append(sample)
        
        # Generate samples for other languages
        for lang in other_languages:
            logger.info(f"Generating {samples_per_other_lang} {lang} samples...")
            for _ in tqdm(range(samples_per_other_lang), desc=f"{lang} samples"):
                sample = self.generate_sample(lang)
                if sample:
                    all_samples.append(sample)
        
        # Shuffle samples
        random.shuffle(all_samples)
        
        # Create splits
        train_size = self.config['dataset']['train_size']
        val_size = self.config['dataset']['val_size']
        test_size = self.config['dataset']['test_size']
        
        # Ensure we have enough samples
        total_needed = train_size + val_size + test_size
        if len(all_samples) < total_needed:
            logger.warning(f"Not enough samples generated. Got {len(all_samples)}, need {total_needed}")
            # Generate more samples if needed
            additional_needed = total_needed - len(all_samples)
            for _ in range(additional_needed):
                lang = random.choice(['he'] + other_languages)
                sample = self.generate_sample(lang)
                if sample:
                    all_samples.append(sample)
        
        # Split data
        train_samples = all_samples[:train_size]
        val_samples = all_samples[train_size:train_size + val_size]
        test_samples = all_samples[train_size + val_size:train_size + val_size + test_size]
        
        # Convert to datasets
        train_dataset = Dataset.from_pandas(pd.DataFrame(train_samples))
        val_dataset = Dataset.from_pandas(pd.DataFrame(val_samples))
        test_dataset = Dataset.from_pandas(pd.DataFrame(test_samples))
        
        # Create dataset dict
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        logger.info(f"Dataset created successfully!")
        logger.info(f"Train: {len(train_dataset)} samples")
        logger.info(f"Validation: {len(val_dataset)} samples")
        logger.info(f"Test: {len(test_dataset)} samples")
        
        return dataset_dict


class PIIDataProcessor:
    """
    Process and tokenize data for model training.
    
    This class handles the tokenization of text and alignment of labels
    with subword tokens, which is crucial for BERT-like models.
    """
    
    def __init__(self, tokenizer, max_length: int = 128):
        """
        Initialize data processor.
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_id = {
            "O": 0,
            "B-PII": 1,
            "I-PII": 2
        }
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
    
    def tokenize_and_align_labels(self, examples):
        """
        Tokenize text and align labels with subword tokens.
        
        This function handles the complex task of aligning word-level labels
        with subword tokens produced by the tokenizer.
        
        Args:
            examples: Batch of examples from dataset
            
        Returns:
            Tokenized inputs with aligned labels
        """
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            padding=False,  # Let data collator handle padding
            max_length=self.max_length,
            is_split_into_words=True,
            return_offsets_mapping=True
        )
        
        labels = []
        for i, label_list in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    # Special token (CLS, SEP, PAD)
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # First token of a word
                    label_ids.append(self.label_to_id[label_list[word_idx]])
                else:
                    # Other tokens of a word
                    # For I-PII, continue with I-PII; otherwise O
                    if label_list[word_idx] == "I-PII":
                        label_ids.append(self.label_to_id["I-PII"])
                    elif label_list[word_idx] == "B-PII":
                        label_ids.append(self.label_to_id["I-PII"])
                    else:
                        label_ids.append(self.label_to_id["O"])
                
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        # Remove offset mapping as it's not needed for training
        tokenized_inputs.pop("offset_mapping")
        
        return tokenized_inputs


def main():
    """Test dataset creation"""
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Create dataset
    dataset_creator = MultilingualPIIDataset(config)
    
    # Generate a few samples for testing
    logger.info("Generating sample data...")
    
    # Test Hebrew sample
    hebrew_sample = dataset_creator.generate_sample('he')
    logger.info(f"Hebrew sample:")
    logger.info(f"Text: {hebrew_sample['text']}")
    logger.info(f"Tokens: {hebrew_sample['tokens']}")
    logger.info(f"Labels: {hebrew_sample['labels']}")
    
    # Test English sample
    english_sample = dataset_creator.generate_sample('en')
    logger.info(f"\nEnglish sample:")
    logger.info(f"Text: {english_sample['text']}")
    logger.info(f"Tokens: {english_sample['tokens']}")
    logger.info(f"Labels: {english_sample['labels']}")
    
    # Create small dataset for testing
    logger.info("\nCreating small test dataset...")
    small_dataset = dataset_creator.create_dataset(100)
    logger.info(f"Dataset info: {small_dataset}")


if __name__ == "__main__":
    main()
