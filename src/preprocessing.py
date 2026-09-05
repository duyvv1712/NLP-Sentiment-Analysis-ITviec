import re
import os
import unicodedata
import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple
from underthesea import word_tokenize

class TextPreprocessor:
    """
    Class xử lý tiền xử lý văn bản tiếng Việt & trích xuất đặc trưng Lexicon cho Sentiment Analysis.
    """
    def __init__(self, dict_dir: str = None):
        if dict_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            dict_dir = os.path.join(os.path.dirname(current_dir), 'data', 'dictionaries')
        
        self.dict_dir = dict_dir
        self.stopwords = self._load_set_from_file(os.path.join(dict_dir, 'vietnamese-stopwords.txt'))
        self.teencode_dict = self._load_dict_from_file(os.path.join(dict_dir, 'teencode.txt'))
        self.wrong_words_dict = self._load_dict_from_file(os.path.join(dict_dir, 'wrong-word.txt'))
        self.english_vnmese_dict = self._load_dict_from_file(os.path.join(dict_dir, 'english-vnmese.txt'))
        self.emoji_dict = self._load_dict_from_file(os.path.join(dict_dir, 'emojicon.txt'))
        
        # Load lexicon cảm xúc
        self.positive_words = self._load_set_from_file(os.path.join(dict_dir, 'positive_words.txt'))
        self.negative_words = self._load_set_from_file(os.path.join(dict_dir, 'negative_words.txt'))
        self.positive_emojis = self._load_set_from_file(os.path.join(dict_dir, 'positive_emoji.txt'))
        self.negative_emojis = self._load_set_from_file(os.path.join(dict_dir, 'negative_emoji.txt'))

        # Chuẩn hóa cụm từ cảm xúc & xác định độ dài cụm từ tối đa cho Greedy Longest Matching
        self.positive_phrases = {
            re.sub(r'\s+', ' ', p.replace('_', ' ')).strip()
            for p in self.positive_words if p.strip()
        }
        self.negative_phrases = {
            re.sub(r'\s+', ' ', p.replace('_', ' ')).strip()
            for p in self.negative_words if p.strip()
        }
        all_phrases = self.positive_phrases | self.negative_phrases
        self.max_phrase_len = max((len(p.split()) for p in all_phrases), default=1)

    def _load_set_from_file(self, filepath: str) -> Set[str]:
        if not os.path.exists(filepath):
            return set()
        with open(filepath, 'r', encoding='utf-8') as f:
            words = set()
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    words.add(unicodedata.normalize('NFC', line.lower()))
            return words

    def _load_dict_from_file(self, filepath: str) -> Dict[str, str]:
        if not os.path.exists(filepath):
            return {}
        result = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    result[parts[0].lower()] = parts[1].lower()
                elif len(parts) == 1 and ' ' in line:
                    subparts = line.strip().split(' ', 1)
                    if len(subparts) == 2:
                        result[subparts[0].lower()] = subparts[1].lower()
        return result

    def normalize_unicode(self, text: str) -> str:
        """Chuẩn hóa bảng mã Unicode sang chuẩn NFC."""
        if not isinstance(text, str):
            return ""
        return unicodedata.normalize('NFC', text)

    def process_emojis(self, text: str) -> str:
        """Thay thế emoji/emojicon bằng từ ngữ mang sắc thái cảm xúc."""
        for emo, replacement in self.emoji_dict.items():
            text = text.replace(emo, f" {replacement} ")
        for emo in self.positive_emojis:
            text = text.replace(emo, " tích_cực ")
        for emo in self.negative_emojis:
            text = text.replace(emo, " tiêu_cực ")
        return text

    def replace_teencode_and_typos(self, text: str) -> str:
        """Thay thế viết tắt, teencode, thuật ngữ IT và lỗi chính tả."""
        # Xử lý các cụm teencode có khoảng trắng trước
        for k, v in self.teencode_dict.items():
            if ' ' in k and k in text.lower():
                pattern = re.compile(re.escape(k), re.IGNORECASE)
                text = pattern.sub(v, text)

        words = text.split()
        normalized_words = []
        for word in words:
            w_lower = word.lower()
            if w_lower in self.teencode_dict:
                normalized_words.append(self.teencode_dict[w_lower])
            elif w_lower in self.wrong_words_dict:
                normalized_words.append(self.wrong_words_dict[w_lower])
            elif w_lower in self.english_vnmese_dict:
                normalized_words.append(self.english_vnmese_dict[w_lower])
            else:
                normalized_words.append(word)
        return " ".join(normalized_words)

    def clean_basic_text(self, text: str) -> str:
        """
        Bước 1: Làm sạch cơ bản (Clean Basic Text)
        - Chuẩn hóa Unicode NFC
        - Xử lý emoji & biểu tượng cảm xúc
        - Chuẩn hóa teencode, từ tiếng Anh, từ sai chính tả
        - Xóa liên kết URL, email, ký tự đặc biệt vô nghĩa
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        text = self.normalize_unicode(text)
        text = self.process_emojis(text)
        
        # Xóa URL và email
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Chuyển về chữ thường
        text = text.lower()
        
        # Thay thế teencode & từ sai
        text = self.replace_teencode_and_typos(text)
        
        # Giữ lại các chữ cái tiếng Việt, số và khoảng trắng
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def clean_advance_text(self, text: str, remove_stopwords: bool = True) -> str:
        """
        Bước 2: Làm sạch nâng cao (Clean Advance Text)
        - Thực hiện tách từ tiếng Việt (Word Segmentation) bằng underthesea
        - Loại bỏ từ dừng (Stopwords) nếu yêu cầu
        """
        basic = self.clean_basic_text(text)
        if not basic:
            return ""
        
        tokenized = word_tokenize(basic, format="text")
        
        if remove_stopwords:
            words = tokenized.split()
            words = [w for w in words if w not in self.stopwords]
            return " ".join(words)
        
        return tokenized

    def calc_sentiment_features(self, text: str, raw_text: str = None) -> Dict[str, float]:
        """
        Trích xuất các thuộc tính thống kê Lexicon bằng thuật toán Greedy Longest Phrase Matching:
        - pos_w, neg_w: Số cụm từ tích cực / tiêu cực (ưu tiên cụm dài nhất trước)
        - pos_e, neg_e: Số emoji tích cực / tiêu cực
        - total_we: Tổng số từ & emoji mang cảm xúc
        - sentiment_ratio: Tỷ lệ cân bằng giữa tích cực và tiêu cực
        """
        if not isinstance(text, str) or not text.strip():
            return {
                'pos_w': 0, 'neg_w': 0, 'pos_e': 0, 'neg_e': 0,
                'total_we': 0, 'sentiment_ratio': 0.0
            }
        
        # Chuẩn hóa văn bản: Hỗ trợ cả text sau word_tokenize (có dấu _) lẫn text thường
        clean_text = unicodedata.normalize('NFC', text.lower()).replace('_', ' ')
        clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
        tokens = clean_text.split()
        
        pos_w = 0
        neg_w = 0
        i = 0
        n = len(tokens)
        
        while i < n:
            matched = False
            # Quét cửa sổ n-gram từ độ dài lớn nhất giảm dần về 1
            for k in range(min(self.max_phrase_len, n - i), 0, -1):
                phrase = " ".join(tokens[i : i + k])
                if phrase in self.positive_phrases:
                    pos_w += 1
                    i += k
                    matched = True
                    break
                elif phrase in self.negative_phrases:
                    neg_w += 1
                    i += k
                    matched = True
                    break
            if not matched:
                i += 1
        
        # Đếm emoji: ưu tiên chuỗi raw_text nếu có, hoặc đếm trực tiếp từ text
        search_text = raw_text if isinstance(raw_text, str) and raw_text.strip() else text
        pos_e = sum(search_text.count(e) for e in self.positive_emojis)
        neg_e = sum(search_text.count(e) for e in self.negative_emojis)
        
        # Bắt thêm các nhãn emoji đã qua xử lý nếu có
        if pos_e == 0 and 'tích_cực' in text:
            pos_e = text.count('tích_cực')
        if neg_e == 0 and 'tiêu_cực' in text:
            neg_e = text.count('tiêu_cực')
        
        total_w = pos_w + neg_w
        total_e = pos_e + neg_e
        total_we = total_w + total_e
        
        # Tỷ lệ cảm xúc chuẩn hóa từ -1 (tiêu cực) đến +1 (tích cực)
        if total_we > 0:
            sentiment_ratio = (pos_w + pos_e - neg_w - neg_e) / float(total_we)
        else:
            sentiment_ratio = 0.0
            
        return {
            'pos_w': pos_w,
            'neg_w': neg_w,
            'pos_e': pos_e,
            'neg_e': neg_e,
            'total_we': total_we,
            'sentiment_ratio': round(sentiment_ratio, 4)
        }

    @staticmethod
    def map_sentiment_label(rating: int) -> str:
        """
        Gán nhãn cảm xúc dựa trên số sao đánh giá (Rating):
        - 4 hoặc 5 sao -> 'Positive' (Tích cực)
        - 3 sao        -> 'Neutral'  (Trung tính)
        - 1 hoặc 2 sao -> 'Negative' (Tiêu cực)
        """
        if rating >= 4:
            return 'Positive'
        elif rating == 3:
            return 'Neutral'
        else:
            return 'Negative'
