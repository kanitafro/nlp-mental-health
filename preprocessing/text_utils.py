from IPython.utils.text import string

import re
from nltk.corpus import stopwords
import nltk
import string
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab') 

# -----------------------------------------------------
# 0️⃣ Stopwords (keep pronouns)
# -----------------------------------------------------
STOPWORDS = set(stopwords.words('english'))
KEEP_WORDS = {
    "i", "me", "my", "mine",
    "not", "no", "nor",
    "don’t", "don't", "didn’t", "didn't",
    "doesn’t", "doesn't", "won’t", "won't",
    "can’t", "can't", "couldn’t", "couldn't",
    "wouldn’t", "wouldn't", "shouldn’t", "shouldn't",
    "ain’t", "aint"
}
#STOPWORDS = STOPWORDS - KEEP_WORDS

NEGATION_WORDS = {
    "no", "not", "never", "none", "nobody", "nothing", "neither", "nowhere",
    "hardly", "scarcely", "barely", "don’t", "doesn’t", "isn’t", "wasn’t",
    "shouldn’t", "wouldn’t", "couldn’t", "won’t", "can’t", "didn’t"
}

# -----------------------------------------------------
# Missing apostrophes
# -----------------------------------------------------
MISSING_APOSTROPHE_MAP = {
    "im": "i'm", "ive": "i've", "id": "i'd", "ill": "i'll",
    "dont": "don't", "didnt": "didn't", "doesnt": "doesn't",
    "isnt": "isn't", "wasnt": "wasn't", "werent": "weren't",
    "havent": "haven't", "hasnt": "hasn't", "hadnt": "hadn't",
    "wont": "won't", "wouldnt": "wouldn't", "cant": "can't",
    "couldnt": "couldn't", "shouldnt": "shouldn't", "mightnt": "mightn't",
    "mustnt": "mustn't", "shes": "she's", "hes": "he's",
    "theyre": "they're", "youre": "you're", "weve": "we've",
    "thats": "that's", "theres": "there's", "heres": "here's",
    "whats": "what's", "whos": "who's", "howre": "how're",
    "lets": "let's", "yall": "y'all"
}

# -----------------------------------------------------
# 2️⃣ Contractions
# -----------------------------------------------------
CONTRACTION_MAP = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
    "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
    "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have", "he's": "he is",
    "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
    "i'd": "i would", "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have",
    "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
    "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not",
    "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
    "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
    "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
    "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
    "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
    "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have",
    "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
    "so's": "so is", "that'd": "that would", "that'd've": "that would have", "that's": "that is",
    "there'd": "there would", "there'd've": "there would have", "there's": "there is",
    "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
    "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
    "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
    "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
    "what'll": "what will", "what'll've": "what will have", "what're": "what are",
    "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have",
    "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will",
    "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is",
    "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
    "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
    "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
    "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
    "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
    "you're": "you are", "you've": "you have"
}

# -----------------------------------------------------
# 3️⃣ Slang / SMS abbreviations
# -----------------------------------------------------
SLANG_MAP = {
    "u": "you", "ur": "your", "r": "are", "ya": "you", "ure": "you are", "yo": "hello",
    "bc": "because", "bcs": "because", "cya": "see you", "cu": "see you",
    "brb": "be right back", "btw": "by the way", "omg": "oh my god", "w/": "with", "w/o": "without",
    "idk": "i do not know", "idc": "i do not care", "imo": "in my opinion",
    "imho": "in my humble opinion", "irl": "in real life", "jk": "just kidding",
    "np": "no problem", "nvm": "never mind", "omw": "on my way", "thx": "thanks",
    "ty": "thank you", "tysm": "thank you so much", "plz": "please", "pls": "please",
    "k": "okay", "kk": "okay", "okie": "okay", "ok": "okay", "wanna": "want to", "gonna": "going to",
    "gotta": "got to", "innit": "isn't it", "lemme": "let me", "gimme": "give me",
    "wth": "what the hell", "wtf": "what the hell", "lmao": "laughing my ass off",
    "lol": "laughing out loud", "rofl": "rolling on the floor laughing", "smh": "shaking my head",
    "tbh": "to be honest", "b4": "before", "gr8": "great", "2day": "today",
    "2moro": "tomorrow", "l8r": "later", "luv": "love", "bday": "birthday",
    "bf": "boyfriend", "gf": "girlfriend", "fam": "family", "sis": "sister", "rly": "really",
    "bro": "brother", "bcuz": "because", "tho": "though", "sry": "sorry", "srsly": "seriously",
    "xoxo": "hugs and kisses", "wyd": "what are you doing", "wya": "where are you at",
    "hru": "how are you", "idgaf": "i do not care", "afaik": "as far as i know",
    "fyi": "for your information", "tldr": "too long didn't read", "msg": "message",
    "dm": "direct message", "pic": "picture", "ppl": "people", "bff": "best friend forever",
    "fr": "for real", "b4n": "bye for now", "ttyl": "talk to you later",
    "ily": "i love you", "ily2": "i love you too", "ikr": "i know right",
    "omfg": "oh my god", "rn": "right now", "asap": "as soon as possible",
    "atm": "at the moment", "fomo": "fear of missing out", "nsfw": "not safe for work",
    "tbf": "to be fair", "obv": "obviously", "tfw": "that feeling when", "smth": "something"
}

# -----------------------------------------------------
# Emoji Handling
# -----------------------------------------------------
EMOJI_LITERAL_MAP = {
    "😀": "happy", "😃": "happy", "😄": "happy", "😁": "happy", "😆": "laughing",
    "😂": "laughing", "🤣": "laughing", "😊": "smiling", "🙂": "content", "🙃": "playful",
    "😉": "wink", "😍": "love", "😘": "love", "😗": "kiss", "😙": "kiss", "😚": "kiss",
    "❤️": "love", "❤": "love", "🧡": "love", "💛": "love", "💚": "love", "💙": "love",
    "💜": "love", "🖤": "love", "💔": "heartbreak", "💖": "love", "💞": "affection",
    "💘": "love", "💝": "love", "😢": "sad", "😭": "crying", "😞": "sad", "😔": "sad",
    "😟": "worried", "😕": "confused", "🙁": "sad", "☹️": "sad", "😣": "frustrated",
    "😖": "upset", "😫": "tired", "😩": "tired", "😤": "angry", "😠": "angry", "😡": "angry",
    "🤬": "furious", "😨": "fear", "😰": "fear", "😱": "terrified", "😳": "embarrassed",
    "😬": "awkward", "😐": "neutral", "😑": "neutral", "😶": "neutral", "😇": "grateful",
    "🙏": "prayer", "💪": "strong", "🤞": "hopeful", "🤷": "uncertain", "🤔": "thinking",
    "😴": "sleepy", "💤": "sleepy", "🤒": "sick", "🤕": "injured", "🤢": "disgust",
    "🤮": "disgust", "😷": "sick", "🤧": "sick", "🤯": "surprised", "😲": "surprised",
    "😯": "surprised", "😮": "surprised", "😏": "flirty", "😒": "unimpressed",
    "🙄": "annoyed", "😌": "relieved", "💀": "death", "☠️": "death", "😈": "mischievous",
    "👿": "angry", "💩": "bad", "🔥": "excited", "🌈": "hope", "🌹": "love", "🌻": "positive",
    "⭐": "success", "✨": "positive", "🌙": "calm", "☀️": "happy", "🌧️": "sad", "🌩️": "angry"
}

EMOJI_AMBIGUOUS_MAP = {
    "💀": ["death","laughing"],
    "☠️": ["death","laughing"],
    "😭": ["crying","laughing"],
    "🤕": ["injured", "sad"]
}

# Emoticons (text-based faces)
EMOTICON_MAP = {
    ":)": "happy", ":-": "happy", ":D": "laughing", ":-D": "laughing", ":(": "sad", ":-(": "sad",
    ":'(": "crying", ":/": "uncertain", ":-/": "uncertain", ":|": "neutral", ":-|": "neutral",
    ";)": "wink", ";-": "wink", ":P": "playful", ":-P": "playful", ":o": "surprised", ":-o": "surprised",
    "XD": "laughing", "xD": "laughing", ":*": "kiss", ":-*": "kiss", ">:(": "angry"
}

# ---------------------------
# Helper Functions
# ---------------------------
def fix_missing_apostrophes(text: str) -> str:
    """
    Fix common contractions missing apostrophes.
    Example: ive -> i've, dont -> don't
    """
    for c, full in MISSING_APOSTROPHE_MAP.items():
        # Use word boundaries, ignore case
        pattern = re.compile(rf'\b{re.escape(c)}\b', flags=re.IGNORECASE)
        text = pattern.sub(full, text)
    return text

def expand_contractions_and_slang(text: str) -> str:
    """Expand contractions and common slang"""
    tokens = text.split()
    expanded = []
    for token in tokens:
        lower = token.lower()
        if lower in CONTRACTION_MAP:
            expanded.extend(CONTRACTION_MAP[lower].split())
        elif lower in SLANG_MAP:
            expanded.extend(SLANG_MAP[lower].split())
        else:
            expanded.append(token)
    return " ".join(expanded)

def replace_emojis_and_emoticons(text: str, mode="ml", use_ner_tags=False) -> str:
    """Replace emojis and emoticons, handle ambiguous emojis"""
    # Emoticons
    for emo, meaning in EMOTICON_MAP.items():
        if emo in text:
            if use_ner_tags:
                text = text.replace(emo, f"[EMOTION_{meaning.upper()}]")
            else:
                text = text.replace(emo, f" {meaning} ")

    # Emojis
    for emo, meaning in EMOJI_LITERAL_MAP.items():
        if emo in text:
            if emo in EMOJI_AMBIGUOUS_MAP:
                senses = EMOJI_AMBIGUOUS_MAP[emo]
                if use_ner_tags:
                    text = text.replace(emo, "[EMOJI]")
                elif mode == "ml":
                    text = text.replace(emo, " ".join(senses))
                else:
                    text = text.replace(emo, f"{emo} [EMOJI]")
            else:
                if use_ner_tags:
                    text = text.replace(emo, f"[EMOTION_{meaning.upper()}]")
                else:
                    text = text.replace(emo, f" {meaning} ")
    return text

def handle_negations(tokens):
    """Prefix NOT_ to the first content word after a negation"""
    result = []
    for i, token in enumerate(tokens):
        result.append(token)
        if token.lower() in NEGATION_WORDS and i + 1 < len(tokens):
            # look for next content word
            for j in range(i + 1, len(tokens)):
                nxt = tokens[j]
                if nxt.lower() in KEEP_WORDS or re.match(r"[.!?,;:]", nxt):
                    break
                tokens[j] = "NOT_" + nxt
                break  # only first content word
    return tokens
