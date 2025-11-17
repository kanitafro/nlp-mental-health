
def get_lexicon():
    lexicon = {
        "Mental Health": {
            "Depression": {
                "keywords": [
                    "depressed", "hopeless", "numb", "sadness", "worthless", "empty", "low mood",
                    "crying", "down", "despair", "lonely", "fatigued", "can't get out of bed",
                    "lost interest", "no motivation", "lifeless", "gloomy", "broken", "isolated",
                    "tearful", "heavy heart", "feeling dead inside", "dark thoughts", "no energy",
                    "giving up", "self-loathing", "feeling like a burden", "nothing matters",
                    "emptiness inside", "losing the spark", "losing spark"
                ],
                "emotions": ["sadness", "nervousness", "grief", "disgust"]
            },
            "Anxiety": {
                "keywords": [
                    "anxious", "panic", "nervous", "overthinking", "worry", "stress", "tense",
                    "racing heart", "restless", "fear", "panic attack", "anxiety attack", "unease",
                    "dread", "butterflies", "shortness of breath", "trembling", "fidgeting",
                    "sweaty palms", "fear of failure", "overwhelmed by thoughts", "insomnia",
                    "on edge", "intrusive thoughts", "catastrophizing", "fear of judgment",
                    "can't relax", "hyperventilating"
                ],
                "emotions": ["nervousness", "sadness", "fear", "anger"] # add anxiety after merging datasets
            },
            "Relief": {
                "keywords": [
                    "calm", "at ease", "lighter", "free", "relieved", "peace of mind", "release",
                    "exhale", "weight lifted", "finally okay", "tranquility", "serene", "comforted",
                    "safe again", "no more pressure", "grateful it's over", "feeling better",
                    "unburdened", "peaceful", "relaxed", "letting go", "closure", "found peace"
                ],
                "emotions": ["relief", "gratitude", "optimism"]
            },
            "Resilience": {
                "keywords": [
                    "strong", "overcame", "bounced back", "coping", "persistence", "endurance",
                    "inner strength", "determination", "never gave up", "pushing through", "healing",
                    "growing", "survived", "adapting", "courage", "self-belief", "standing tall",
                    "learned from it", "emotional strength", "grit", "perseverance", "staying hopeful",
                    "rebuilding", "rising again", "heal"
                ],
                "emotions": ["love", "pride", "relief", "optimism"]
            },
            "Stress": {
                "keywords": [
                    "overwhelmed", "pressure", "tension", "burnout", "exhausted", "under pressure",
                    "heavy workload", "stress", "losing my mind", "can't handle it", "tight schedule",
                    "mental strain", "work overload", "deadline anxiety", "frustration", "fatigue",
                    "snapping easily", "emotionally drained", "overworked", "restless nights",
                    "too much going on", "feeling stuck", "tense muscles", "drained energy"
                ],
                "emotions": ["nervousness", "anger", "annoyance", "fear"]
            },
            "Burnout": {
                "keywords": [
                    "drained", "emotionally tired", "can't focus", "exhausted from work",
                    "exhausted from school", "detached", "no motivation left", "running on empty",
                    "mentally done", "numb from exhaustion", "overworked", "sleep deprived",
                    "brain fog", "can't keep up", "tired of everything", "emotional exhaustion",
                    "loss of purpose", "cynical", "depleted", "energy crash", "constant fatigue"
                ],
                "emotions": ["sadness", "annoyance", "neutral"]
            },
            "Coping strategies": {
                "keywords": [
                    "meditation", "journaling", "therapy", "talking to someone", "self-care",
                    "mindfulness", "breathing exercises", "support group", "exercise", "nature walk",
                    "music", "grounding techniques", "relaxation", "healthy routine", "gratitude journal",
                    "art therapy", "prayer", "distraction", "healthy eating", "yoga", "setting boundaries",
                    "digital detox", "time off", "emotional regulation", "reaching out", "deep breathing",
                    "therapy", "psychiatrist", "psychology", "CBT", "DBT", "psychologist"
                ],
                "emotions": ["relief", "caring", "love", "pride", "neutral"]
            },
            "Self-harm risk": {
                "keywords": [
                    "suicidal", "hurt myself", "cutting", "cuts", "scars", "can't go on", "end it",
                    "life not worth living", "want to die", "hopeless", "razor", "overdose",
                    "harming myself", "self-inflicted pain", "emotional pain", "invisible wounds", "desperate",
                    "lost the will to live", "hurting inside", "done with everything", "bruising", "cover up scars"
                ],
                "emotions": ["sadness", "disgust", "disappointment"]
            },
            "Suicidal": {
                "keywords": [
                    "suicidal", "hurt myself", "cut wrists", "cutting", "can't go on", "end it",
                    "life not worth living", "want to die", "hopeless", "razor", "overdose",
                    "harming myself", "ending everything", "tired of life", "wish I wasn't here",
                    "thinking of ending it", "self-inflicted pain", "no reason to live", "empty"
                    "emotional pain", "invisible wounds", "giving up on life", "desperate",
                    "lost the will to live", "hurting inside", "done with everything", "burden"
                    "better off without me", "suicide note", "letter", "note", "jump", "jump off",
                    "lost the will to live", "no will to live", "trapped", "i'm sorry", "hopeless"
                ],
                "emotions": ["sadness", "disgust", "disappointment"] # add suicidal here after merging datasets
            },
            "Substance use": {
                "keywords": [
                    "alcohol", "drunk", "weed", "smoking", "drugs", "high", "addicted", "sober",
                    "relapse", "overdose", "drinking problem", "intoxicated", "hangover", "withdrawal",
                    "rehab", "temptation", "dependency", "craving", "substance abuse", "getting wasted",
                    "losing control", "numbing the pain", "relapse risk", "clean for a week",
                    "under the influence", "binge drinking", "staying clean"
                ],
                "emotions": ["sadness", "disgust", "disappointment", "fear", "anger", "disapproval"]
            },
            "Risk-taking": {
                "keywords": [
                    "reckless", "impulsive", "danger", "thrill", "risk", "adrenaline", "out of control",
                    "sudden urge", "taking chances", "no fear", "living on edge", "risky behavior",
                    "daredevil", "acting without thinking", "doing something crazy", "breaking rules",
                    "chasing excitement", "spontaneous", "testing limits", "high-risk",
                    "impulsive decisions", "thrill-seeking", "not thinking straight", "adrenaline rush"
                ],
                "emotions": ["fear", "anger", "curiosity", "excitement"]
            }
        },
        "Grief & Loss": {
            "Grief": {
                "keywords": [
                    "loss", "gone", "died", "pass away", "funeral", "grief", "heartbreak", "mourning",
                    "deep sorrow", "devastated", "shattered", "broken heart", "emptiness", "heavy heart",
                    "crying", "inconsolable", "sadness", "numb", "sorrowful", "bereft", "tragedy",
                    "pain", "anguish", "heartache", "feeling lost", "emotional pain",
                    "mourning a loved one", "life feels empty", "unbearable sadness"
                ],
                "emotions": ["sadness", "grief", "remorse", "love"]
            },
            "Mourning": {
                "keywords": [
                    "remembrance", "crying", "bereavement", "condolences", "miss you", "tribute",
                    "memorial", "vigil", "paying respects", "honoring", "reflection", "sorrow",
                    "lamenting", "grieving together", "sympathy", "shared loss", "flowers for the deceased",
                    "sending thoughts", "comforted in sorrow", "attending funeral", "loss ritual",
                    "keeping memory alive", "deceased"
                ],
                "emotions": ["grief", "sadness"]
            },
            "Nostalgia": {
                "keywords": [
                    "childhood", "memories", "old days", "past", "remember when", "flashback",
                    "days gone by", "reminisce", "sentimental", "good old times", "longing for the past",
                    "family moments", "old friends", "youth", "simpler times", "cherished memories",
                    "photo albums", "storytelling", "sentimental journey", "bittersweet feelings",
                    "recalling laughter", "warm memories", "recalling traditions"
                ],
                "emotions": ["sadness", "confusion", "neutral"]
            },
            "Missing someone": {
                "keywords": [
                    "miss you", "wish you were here", "absence", "long for", "lonely without you",
                    "you're in my heart", "still in our hearts", "still in my heart", "thinking of you",
                    "counting the days", "feeling empty", "longing", "can't stop thinking about you",
                    "separation pain", "wish you were near", "craving your presence", "distance hurts",
                    "wishing you were back", "left behind", "yearning", "thinking of old times", "lonely"
                ],
                "emotions": ["sadness", "grief", "love"]
            },
            "Loneliness": {
                "keywords": [
                    "lonely", "alone", "isolated", "nobody", "empty room", "no one to talk to",
                    "solitude", "abandoned", "friendless", "quiet", "left out", "disconnected",
                    "feeling invisible", "longing for company", "solitude weighs", "solitary",
                    "empty heart", "hollow", "confinement", "craving connection", "feeling left behind",
                    "yearning for companionship", "solitude pain", "isolation"
                ],
                "emotions": ["sadness", "grief"]
            },
            "Regret": {
                "keywords": [
                    "should have", "if only", "sorry for", "wish I hadn't", "remorse", "guilt",
                    "hindsight", "missed opportunity", "could've done better", "self-blame",
                    "feeling responsible", "shame", "wishing things were different", "apology",
                    "making amends", "dwelling on mistakes", "regrets weigh heavy", "hindsight hurts",
                    "wish I could turn back time", "second-guessing", "feeling accountable",
                    "repentance", "rueful"
                ],
                "emotions": ["remorse", "sadness", "grief", "disappointment", "embarrassment", "disgust"]
            }
        },

        "SPIRITUALITY": {
            "Faith": {
                "keywords": [
                    "soul", "spirit", "higher power", "inner peace", "divine", "connection", "faith",
                    "universe", "belief", "trust in God", "devotion", "spirituality", "karma", "dharma",
                    "samsara", "enlightenment", "awakening", "Atman", "Brahman", "bodhi", "mindfulness",
                    "sacred", "transcendence", "spiritual journey", "surrender", "grace", "sacred path",
                    "cosmic energy", "sacred presence"
                ],
                "emotions": ["gratitude", "love", "remorse", "relief", "admiration", "joy", "curiosity"]
            },
            "Prayer": {
                "keywords": [
                    "pray", "praying", "dua", "thank God", "gratitude", "asking for guidance",
                    "grace", "salah", "isha", "maghrib", "fajr", "dhuhr", "asr", "mantra", "chanting",
                    "meditation", "puja", "offering", "prostration", "hymn", "kirtan", "recitation",
                    "rosary", "confession", "invocation", "liturgy", "contemplative prayer",
                    "silent prayer", "mindful prayer", "guided meditation", "spiritual petition",
                    "lighting incense", "prayer beads"
                ],
                "emotions": ["gratitude", "love", "remorse", "relief", "admiration"]
            },
            "Meaning-making": {
                "keywords": [
                    "purpose", "meaning", "destiny", "why am I here", "life path", "calling",
                    "life lesson", "spiritual growth", "dharma", "moksha", "karma", "self-realization",
                    "enlightenment", "awakening", "soul mission", "divine plan", "cosmic purpose",
                    "seeking truth", "ultimate reality", "inner guidance", "life questions",
                    "existential inquiry", "finding God", "path to Nirvana", "discovering inner self"
                ],
                "emotions": ["confusion", "curiosity", "love", "surprise", "optimism", "realization", "nervousness", "surprise", "fear"]
            },
            "Doubt": {
                "keywords": [
                    "questioning", "uncertainty", "disbelief", "lost faith", "confused about beliefs",
                    "spiritual crisis", "struggling with belief", "doubt in God", "skeptical",
                    "questioning purpose", "wavering faith", "feeling lost spiritually",
                    "lack of guidance", "feeling disconnected from divine", "confusion about dharma",
                    "existential doubt", "moral uncertainty", "seeking clarity", "soul unrest",
                    "spiritual questioning", "struggling with scripture", "faith shaken"
                ],
                "emotions": ["curiosity", "nervousness", "annoyance", "disappointment", "confusion", "sadness"]
            },
            "Purpose": {
                "keywords": [
                    "calling", "goal", "meaning of life", "what I'm meant to do", "life purpose",
                    "vision", "vocation", "life mission", "spiritual aim", "soul work", "dharma",
                    "path of righteousness", "contribution", "higher calling", "destiny",
                    "reason for existence", "self-actualization", "true path", "life direction",
                    "awakening", "fulfilling potential", "inner purpose", "cosmic plan", "sacred duty"
                ],
                "emotions": ["curiosity", "nervousness", "realization", "excitement", "surprise", "neutral"]
            },
            "Mortality": {
                "keywords": [
                    "death", "dying", "awareness of death", "time is short", "impermanence",
                    "fleeting life", "transient", "mortality awareness", "cycle of life and death",
                    "memento mori", "anicca", "rebirth", "samsara", "finality", "end of life",
                    "legacy", "remembering mortality", "accepting death", "preparing for afterlife",
                    "finite existence", "impermanent nature", "spiritual mortality", "mortality reflection"
                ],
                "emotions": ["sadness", "grief", "remorse", "grief", "curiosity", "neutral"]
            },
            "Religion": {
                "keywords": [
                    "church", "mosque", "temple", "synagogue", "Bible", "Qur'an", "Torah", "scripture",
                    "religion", "faith", "God", "Allah", "Jehovah", "Lord", "Christ", "Jesus", "psalm",
                    "surah", "verse", "hadith", "gospel", "holy text", "Buddha", "dharma", "sangha",
                    "meditation hall", "monastery", "temple rituals", "Vedas", "Upanishads",
                    "Bhagavad Gita", "Shiva", "Vishnu", "Krishna", "Hinduism", "Buddhism", "deity",
                    "guru", "spiritual teacher", "prayer wheel", "mandala", "Ar-Rahman", "Ar-Rahim",
                    "Al-Malik", "Al-Quddus", "As-Salam", "Al-Mu'min", "Al-Muhaymin", "Al-Aziz",
                    "Al-Jabbar", "Al-Mutakabbir", "Al-Khaliq", "Al-Bari'", "Al-Musawwir", "Al-Ghaffar",
                    "Al-Qahhar", "Al-Wahhab", "Ar-Razzaq", "Al-Fattah", "Al-'Alim", "Al-Qabid",
                    "Al-Basit", "Al-Khafid", "Ar-Rafi'", "Al-Mu'izz", "Al-Mudhill", "As-Sami'",
                    "Al-Basir", "Al-Hakam", "Al-'Adl", "Al-Latif", "Al-Khabir", "Al-Halim", "Al-Azim",
                    "Al-Ghafur", "Ash-Shakur", "Al-Ali", "Al-Kabir", "Al-Hafiz", "Al-Muqit", "Al-Hasib",
                    "Al-Jalil", "Al-Karim", "Ar-Raqib", "Al-Mujib", "Al-Wasi'", "Al-Hakim", "Al-Wadud",
                    "Al-Majid", "Al-Ba'ith", "Ash-Shahid", "Al-Haqq", "Al-Wakil", "Al-Qawi", "Al-Matin",
                    "Al-Wali", "Al-Hamid", "Al-Muhsi", "Al-Mubdi'", "Al-Mu'id", "Al-Muhyi", "Al-Mumit",
                    "Al-Hayy", "Al-Qayyum", "Al-Wajid", "Al-Majid", "Al-Wahid", "Al-Ahad", "As-Samad",
                    "Al-Qadir", "Al-Muqtadir", "Al-Muqaddim", "Al-Mu'akhkhir", "Al-Awwal", "Al-Akhir",
                    "Az-Zahir", "Al-Batin", "Al-Wali", "Al-Muta'ali", "Al-Barr", "At-Tawwab",
                    "Al-Muntaqim", "Al-'Afuww", "Ar-Ra'uf", "Malik-ul-Mulk", "Dhul-Jalal wal-Ikram",
                    "Al-Muqsit", "Al-Jami'", "Al-Ghani", "Al-Mughni", "Al-Mani'", "Ad-Darr", "An-Nafi'",
                    "An-Nur", "Al-Hadi", "Al-Badi'", "Al-Baqi", "Al-Warith", "Ar-Rashid", "As-Sabur"
                ],
                "emotions": ["gratitude", "love", "relief", "remorse", "neutral", "joy"]
            }
        },
        "RELATIONSHIPS & SOCIAL LIFE": {
            "Family": {
                "keywords": [
                    "mother", "father", "parents", "siblings", "family", "arguments", "upbringing",
                    "home", "family dinner", "guests", "child", "kid", "son", "daughter",
                    "mother-in-law", "father-in-law", "grandma", "grandmother", "grandpa",
                    "grandfather", "grandparents", "step-family", "cousin", "aunt", "uncle",
                    "sibling rivalry", "household", "parent-child bonding", "family traditions",
                    "chores", "family gathering", "parenting", "parental advice", "adoption",
                    "family reunion", "mom", "dad", "brother", "sister", "stepmother", "stepfather", "stepsister",
                    "stepbrother", "guardian", "foster family", "godparent", "niece", "nephew", "relatives", "kin",
                    "ancestors", "descendants", "heirloom", "family home", "family car", "childhood home",
                    "nuclear family", "extended family", "blended family", "single parent", "only child",
                    "family rules", "allowance", "bedtime", "curfew", "family meeting", "game night",
                    "Sunday roast", "holiday meal", "birthday party", "anniversary", "funeral", "wedding",
                    "family vacation", "road trip", "photo album", "home video", "family story", "family tree",
                    "legacy", "roots", "heritage", "culture", "family name", "values", "expectations",
                    "pressure", "favoritism", "black sheep", "golden child", "middle child", "oldest", "youngest",
                    "empty nest", "boomerang kid", "caregiver", "elderly parents", "inheritance", "will",
                    "family business", "family support", "family conflict", "dysfunction", "reconciliation",
                    "unconditional love", "tough love", "tension", "peace", "harmony", "chaos"
                ],
                "emotions": ["love", "caring", "disapproval", "approval", "nervousness", "relief", "joy", "anger", "sadness", "gratitude", "disappointment"]
            },
            "Friendship": {
                "keywords": [
                    "friend", "best friend", "hang out", "trust", "companionship", "betrayal", "hug",
                    "buddy", "pal", "mate", "friendship goals", "loyal friend", "confidant", "peer",
                    "squad", "social circle", "inside joke", "support system", "gossip", "outing",
                    "road trip", "sleepover", "friend request", "messaging", "group chat", "club",
                    "teammate", "acquaintance", "close friend", "childhood friend", "school friend", "work friend",
                    "online friend", "pen pal", "BFF", "ride or die", "partner in crime", "shoulder to cry on",
                    "trust issues", "drift apart", "falling out", "reconciliation", "forgiveness", "apology",
                    "loyalty", "reliability", "dependability", "frenemy", "toxic friendship", "friendship breakup",
                    "platonic love", "brotherhood", "sisterhood", "solidarity", "team spirit",
                    "double date", "night out", "coffee date", "study session", "gym buddy", "travel companion",
                    "venting", "advice", "celebration", "congratulations", "sympathy", "encouragement",
                    "friendiversary", "memory", "shared experience", "common interest", "mutual friend",
                    "social media friends", "ignore", "new friend", "old friend", "long-distance friendship",
                    "growing apart", "jealousy", "competition", "comparison", "third wheel"
                ],
                "emotions": ["love", "caring", "admiration", "gratitude", "neutral", "approval", "grief", "sadness"]
            },
            "Love life": {
                "keywords": [
                    "love", "crush", "boyfriend", "girlfriend", "heartbreak", "date", "relationship",
                    "intimacy", "break up", "partner", "romantic", "soulmate", "passion", "affection",
                    "flirting", "flirty", "proposal", "engagement", "marriage", "wedding", "wife", "husband",
                    "spouse", "commitment", "jealousy", "longing", "lust", "cuddle", "anniversary",
                    "romantic gesture", "love letter", "secret admirer", "unrequited love", "significant other",
                    "partner", "lover", "sweetheart", "darling", "better half", "first date", "blind date", "hookup",
                    "casual dating", "serious relationship", "exclusive", "one-night stand",
                    "situationship", "long-distance relationship", "open relationship", "polyamory",
                    "infatuation", "puppy love", "chemistry", "spark", "connection", "butterflies",
                    "honeymoon phase", "settling down", "moving in", "sex", "meet the parents", "growing old together",
                    "separation", "divorce", "split", "dumped", "ghosted", "cheating", "affair", "unfaithful",
                    "reconciliation", "second chance", "on-again-off-again", "toxic relationship", "codependency",
                    "arguments", "makeup sex", "couples therapy", "love language", "quality time",
                    "kiss", "embrace", "hold hands", "making love", "attraction", "desire", "yearning",
                    "pining", "obsession", "possessiveness", "control", "manipulation", "gaslighting",
                    "trust issues", "vulnerability", "fear of commitment", "fear of abandonment"
                ],
                "emotions": ["love", "caring", "joy", "admiration", "desire", "surprise", "disappointment", "sadness", "gratitude"]
            },
            "Conflict/apology": {
                "keywords": [
                    "argument", "fight", "sorry", "forgive me", "misunderstanding", "resentment",
                    "disagreement", "tension", "blame", "criticism", "harsh words", "conflict resolution",
                    "regret", "insult", "quarrel", "confrontation", "dispute", "reconciliation",
                    "grudges", "yelling", "shouting", "misunderstanding", "anger", "conflict management"
                ],
                "emotions": ["anger", "disappointment", "annoyance", "disapproval", "confusion"]
            },
            "Gratitude": {
                "keywords": [
                    "thankful", "grateful", "appreciation", "love you", "cherish", "blessings",
                    "gratefulness", "indebted", "recognition", "acknowledgment", "praise", "admiration",
                    "respect", "heartfelt thanks", "gratitude note", "compliment", "expressing thanks",
                    "feeling blessed", "showing appreciation", "grateful heart", "thankful for you"
                ],
                "emotions": ["gratitude", "admiration", "caring", "love"]
            },
            "Social support": {
                "keywords": [
                    "support", "help", "advice", "listening", "talking", "comfort", "encouragement",
                    "backup", "guidance", "empathy", "emotional support", "being there",
                    "shoulder to lean on", "mentorship", "cheering up", "consolation", "reassurance",
                    "solidarity", "caring", "checking in", "advocacy", "resource sharing", "hard times"
                ],
                "emotions": ["joy", "gratitude", "caring"]
            },
            "Cultural events": {
                "keywords": [
                    "festivals", "fairs", "parades", "carnivals", "celebrations", "traditions",
                    "holidays", "rituals", "cultural performances", "religious festivals",
                    "music festivals", "art exhibitions", "heritage events", "community ceremonies",
                    "cultural workshops", "storytelling events", "dance performances",
                    "costume festivals", "seasonal festivals", "folk festivals", "traditional celebrations"
                ],
                "emotions": ["joy", "excitement", "amusement"]
            }
        },

        "IDENTITY & SELF-PERCEPTION": {
            "Self-esteem": {
                "keywords": [
                    "confident", "insecure", "proud", "ashamed", "believe in myself", "self-worth",
                    "self-respect", "self-confidence", "self-doubt", "feeling capable", "empowered",
                    "feeling small", "self-assured", "worthy", "inadequate", "self-image", "feeling strong",
                    "feeling weak", "valuing myself", "feeling accomplished", "self-love", "self-acceptance",
                    "feeling inferior", "ego", "feeling validated"
                ],
                "emotions": ["joy", "sadness", "pride", "remorse", "pride", "fear"]
            },
            "Self-discovery": {
                "keywords": [
                    "finding myself", "learning who I am", "personal growth", "introspection", "reflection",
                    "self-awareness", "inner journey", "exploring my identity", "discovering passions",
                    "soul-searching", "understanding emotions", "figuring out my values", "self-realization",
                    "evolving", "growth mindset", "learning life lessons", "spiritual self-discovery",
                    "uncovering strengths", "embracing weaknesses", "questioning beliefs",
                    "defining purpose", "self-exploration"
                ],
                "emotions": ["curiosity", "joy", "confusion", "relief", "surprise", "realization", "excitement"]
            },
            "Gender/identity exploration": {
                "keywords": [
                    "gender identity", "sexuality", "pronouns", "coming out", "queer", "trans", "acceptance",
                    "LGBTQ", "non-binary", "cisgender", "pansexual", "asexual", "gender fluid",
                    "questioning identity", "gender expression", "sexual orientation", "gender dysphoria",
                    "pride", "gender transition", "embracing identity", "exploring identity", "identity affirmation",
                    "self-acceptance", "representation", "inclusivity"
                ],
                "emotions": ["fear", "joy", "pride", "confusion", "love", "nervousness", "neutral"]
            },
            "Boundary-setting": {
                "keywords": [
                    "saying no", "personal space", "limits", "protecting energy", "respect boundaries",
                    "healthy boundaries", "standing up for myself", "setting limits", "emotional boundaries",
                    "asserting needs", "self-protection", "not overcommitting", "saying yes to myself",
                    "preventing burnout", "self-preservation", "prioritizing self", "maintaining autonomy",
                    "stopping manipulation", "saying 'enough'", "respecting others' boundaries", "consent", "assertiveness"
                ],
                "emotions": ["pride", "fear", "relief", "nervousness", "disapproval"]
            },
            "Shame": {
                "keywords": [
                    "embarrassed", "ashamed", "guilty", "regret", "humiliation", "self-blame", "feeling exposed",
                    "mortified", "feeling inadequate", "dishonor", "feeling judged", "self-conscious", "blush",
                    "inner critic", "remorse", "feeling flawed", "disgrace", "feeling dirty", "shame spiral",
                    "humiliation in public", "regretful", "cringe", "self-reproach", "embarrassment"
                ],
                "emotions": ["remorse", "disgust", "sadness", "fear", "anger", "embarrassment"]
            }
    },
    "DESCRIPTIVE & SENSORY WRITING": {
            "Environment/setting": {
                "keywords": [
                    "room", "city", "park", "forest", "ocean", "nature", "surroundings", "street", "alley",
                    "countryside", "mountains", "desert", "garden", "meadow", "beach", "river", "lake",
                    "skyline", "neighborhood", "urban", "rural", "village", "landscape", "horizon",
                    "village square", "bustling market", "abandoned building", "cozy cabin",
                    "open field", "dense jungle", "quiet library", "dimly lit room", "crowded café"
                ],
                "emotions": ["admiration", "fear", "relief", "nervousness", "surprise", "curiosity", "neutral"]
            },
            "Weather/seasonality": {
                "keywords": [
                    "rain", "sunshine", "snow", "autumn", "fall", "spring", "winter", "summer", "storm",
                    "breeze", "sunny", "wind", "drizzle", "downpour", "hail", "thunderstorm", "lightning",
                    "fog", "mist", "overcast", "cloudy", "clear sky", "frost", "icy", "humid", "muggy",
                    "crisp air", "chilly", "warm rays", "golden sunlight", "gentle breeze", "gusty wind",
                    "blustery", "serene weather", "seasonal change", "twilight glow"
                ],
                "emotions": ["joy", "sadness", "fear", "relief", "amusement", "neutral"]
            },
            "Physical sensations": {
                "keywords": [
                        "warm", "cold", "trembling", "heartbeat", "fatigue", "hunger", "touch", "pain",
                        "shivering", "sweating", "tingling", "numb", "itchy", "sore", "aching", "throbbing",
                        "burning", "pressure", "tension", "stiffness", "dizziness", "exhaustion", "relaxation",
                        "comfort", "goosebumps", "fluttering", "pounding", "lightheaded", "sweaty palms",
                        "chills", "electric sensation", "heavy limbs", "soft caress", "prickly"
                ],
                "emotions": ["fear", "joy", "anger", "relief", "nervousness", "disgust"]
            },
            "Aesthetic": {
                "keywords": [
                        "beautiful", "color", "light", "scenery", "art", "music", "poetry", "aesthetic", "vibe",
                        "stunning", "exquisite", "radiant", "dazzling", "enchanting", "vibrant", "delicate",
                        "vivid", "surreal", "whimsical", "harmonious", "soft", "muted", "textured", "intricate",
                        "minimalist", "bold", "ethereal", "moody", "serene", "dramatic", "captivating", "magical",
                        "enchanting", "atmospheric", "dreamy", "blue", "purple", "yellow", "orange", "black",
                        "white", "grey", "periwinkle", "magenta", "cyan", "brown", "green", "red", "turquoise",
                        "indigo", "lilac", "gold", "silver", "bronze", "coral", "beige", "teal", "amber", "cream",
                        "scarlet", "jade", "lavender", "navy", "pastel", "neon", "earthy tones",
                        "muted shades", "vibrant hues"
                ],
                "emotions": ["admiration", "joy", "curiosity", "relief", "excitement", "desire"]
            },
            "Physical descriptions": {
                "keywords": [
                        "tall", "short", "slender", "stocky", "muscular", "thin", "petite", "curvy", "lanky",
                        "broad-shouldered", "lean", "chubby", "graceful", "athletic", "frail", "youthful",
                        "elderly", "wrinkled", "radiant", "glowing", "pale", "tanned", "olive skin",
                        "fair complexion", "dark complexion", "freckles", "moles", "birthmark", "scarred",
                        "elegant", "rugged", "sharp features", "soft features", "angular face", "round face",
                        "oval face", "almond-shaped eyes", "piercing eyes", "sparkling eyes", "hazel eyes",
                        "green eyes", "blue eyes", "brown eyes", "long hair", "short hair", "curly hair",
                        "straight hair", "wavy hair", "messy hair", "dyed hair", "thick hair", "thin hair",
                        "beard", "mustache", "stubble", "expressive eyebrows", "delicate hands", "calloused hands",
                        "graceful posture", "confident stance", "nervous fidgeting", "warm smile", "shy smile", "stern look"
                ],
                "emotions": ["admiration", "fear", "nervousness", "disgust", "curiosity", "desire", "annoyance", "neutral"]
            },
            "Sensory": {
                "keywords": [
                        "silky", "rough", "smooth", "jagged", "sticky", "soft", "hard", "sharp", "bitter",
                        "sweet", "sour", "pungent", "fragrant", "musty", "fresh", "metallic", "earthy",
                        "crisp", "warm", "cool", "humid", "dry", "fragrant", "aromatic", "spicy", "salty",
                        "tangy", "rich", "muted", "dazzling", "shimmering", "glowing", "sparkling", "shadowy",
                        "dark", "luminous", "vibrant", "hazy", "translucent", "opaque"
                ],
                "emotions": ["relief", "disgust", "joy", "fear", "admiration", "annoyance"]
            }
    },

    "LIFESTYLE & LIFE EVENTS": {
            "Career": {
                "keywords": [
                        "job", "interview", "boss", "promotion", "resign", "career path", "work stress", "raise",
                        "fired", "laid off", "new position", "internship", "networking", "project", "team dynamics",
                        "deadlines", "performance review", "career growth", "mentorship", "job offer",
                        "probation period", "workplace conflict", "office politics", "remote work", "commute",
                        "overtime", "contract", "responsibilities", "multitasking", "career pivot",
                        "entrepreneurship", "employee", "employer"
                ],
                "emotions": ["confusion", "disappointment", "pride", "fear", "joy", "nervousness", "optimism", "neutral"]
            },
            "Housing": {
                "keywords": [
                        "apartment", "rent", "moving out", "new city", "roommate", "home", "flat", "house",
                        "mortgage", "landlord", "lease", "neighborhood", "relocation", "furniture",
                        "renovation", "interior design", "moving boxes", "property", "backyard", "garden",
                        "utilities", "decorating", "cozy", "cluttered", "minimalist", "cramped", "spacious",
                        "suburban", "urban", "cityscape", "homely", "temporary housing"
                ],
                "emotions": ["neutral", "excitement", "nervousness", "relief", "fear", "joy"]
            },
            "Finances": {
                "keywords": [
                        "money", "bills", "broke", "saving", "budget", "expenses", "paycheck", "debt", "loan",
                        "interest", "credit card", "bank account", "emergency fund", "financial planning",
                        "investing", "taxes", "rent", "grocery bills", "subscription", "spending habits",
                        "budgeting app", "paycheck-to-paycheck", "wealth", "frugal", "expensive", "luxury",
                        "savings goal", "financial stress", "retirement fund", "managing money"
                ],
                "emotions": ["nervousness", "fear", "relief", "disappointment", "joy"]
            },
            "Health events": {
                "keywords": [
                        "illness", "injury", "diagnosis", "surgery", "recovery", "hospital", "doctor", "clinic",
                        "treatment", "chronic illness", "acute illness", "fever", "medication", "therapy",
                        "check-up", "vaccination", "follow-up", "health scare", "physical therapy",
                        "mental health check", "hospitalization", "side effects", "rehabilitation", "fatigue",
                        "doctor's appointment", "lab tests", "MRI", "X-ray", "procedure", "health insurance",
                        "symptom management", "cancer", "tumor", "hospice", "palliative"
                ],
                "emotions": ["fear", "nervousness", "relief", "grief", "optimism", "sadness", "disgust", "surprise"]
            },
            "Parenthood": {
                "keywords": [
                        "baby", "pregnancy", "parent", "child", "motherhood", "fatherhood", "trying to conceive",
                        "maternal leave", "paternal leave", "first trimester", "second trimester", "third trimester",
                        "trimester", "breastfeeding", "newborn", "toddler", "diaper", "parenting", "adoption",
                        "parenting stress", "bonding", "lullaby", "bedtime routine", "school age", "parenting milestone",
                        "vaccination", "playdate", "family planning", "childcare", "nanny", "parenting advice"
                ],
                "emotions": ["joy", "love", "nervousness", "fear", "caring", "pride", "admiration"]
            },
            "Goals": {
                "keywords": [
                        "goal", "resolution", "plan", "ambition", "checklist", "improvement", "milestone", "aspiration",
                        "objective", "target", "achievement", "personal growth", "challenge", "bucket list",
                        "self-improvement", "vision board", "short-term goal", "long-term goal", "career goal",
                        "fitness goal", "habit goal", "education goal", "learning objective", "motivation",
                        "step-by-step plan", "measurable outcome", "checklist"
                ],
                "emotions": ["curiosity", "excitement", "nervousness", "pride", "fear", "joy", "optimism"]
            },
            "Decision-making": {
                "keywords": [
                        "deciding", "choice", "dilemma", "weighing options", "considering", "pros", "cons",
                        "decision making", "choices", "decision", "uncertainty", "indecision", "pondering",
                        "analyzing", "brainstorming", "evaluating", "risk assessment", "gut feeling",
                        "intuition", "judgment", "prioritizing", "compromise", "selecting", "options", "solution",
                        "alternative", "trade-off", "strategy", "deliberation", "mental weighing"
                ],
                "emotions": ["nervousness", "confusion", "fear", "relief"]
            },
            "Hopes": {
                "keywords": [
                        "dream", "plan ahead", "vision", "next year", "future", "aspirations", "ambition", "hope",
                        "goal-setting", "long-term plan", "next chapter", "life goals", "bucket list", "next steps",
                        "planning", "looking forward", "excitement", "anticipation", "aspirations for family",
                        "travel plans", "retirement plan", "career vision", "personal vision", "optimistic",
                        "foresight", "goal visualization"
                ],
                "emotions": ["excitement", "pride", "joy", "curiosity", "optimism", "fear", "confusion"]
            },
            "Routines": {
                "keywords": [
                        "morning routine", "schedule", "daily habits", "evening routine", "exercise", "skincare",
                        "meditation", "breakfast", "commute", "work routine", "study schedule", "nightly rituals",
                        "cleaning routine", "self-care", "journaling", "yoga", "stretching", "productivity habits",
                        "time management", "habit tracker", "consistent", "habitual", "repetitive tasks",
                        "weekend routine", "meal prep", "bedtime ritual", "fitness regimen"
                ],
                "emotions": ["relief", "neutral", "pride", "nervousness"]
            },
            "Habits": {
                "keywords": [
                        "smoking", "scrolling", "working out", "procrastination", "reading", "journaling",
                        "caffeine intake", "snacking", "drinking", "exercising", "gaming", "watching TV",
                        "checking phone", "meditation", "tidying up", "hobby", "crafting", "social media use",
                        "drinking water", "walking", "mindfulness practice", "sleeping patterns",
                        "alcohol consumption", "impulsive buying", "routine behavior", "habitual action",
                        "addiction", "screen time", "routine habit"
                ],
                "emotions": ["disgust", "pride", "embarrassment", "amusement", "anger", "relief"]
            },
            "Performance/productivity": {
                "keywords": [
                        "efficient", "focus", "task", "deadline", "motivation", "lazy", "productivity",
                        "multitasking", "burnout", "procrastination", "accomplishment", "efficiency",
                        "time management", "output", "workload", "workflow", "organization", "goal completion",
                        "performance metrics", "prioritization", "concentration", "energy", "alertness",
                        "distraction", "deadlines looming", "task-oriented", "accountability",
                        "work ethic", "momentum", "progress", "procrastinate"
                ],
                "emotions": ["fear", "pride", "annoyance", "amusement", "nervousness", "joy", "neutral"]
            },
            "Entertainment": {
                "keywords": [
                        "movies", "films", "TV shows", "series", "binge-watching", "streaming", "concerts",
                        "music festivals", "theater", "performances", "plays", "musicals", "stand-up comedy",
                        "live shows", "online videos", "podcasts", "audiobooks", "radio", "storytelling",
                        "fanfiction", "gaming", "eSports", "board games", "video games",
                        "card games", "cinema", "film reviews"
                ],
                "emotions": ["joy", "excitement", "relief", "amusement", "surprise"]
            },
            "Arts & crafts": {
                "keywords": [
                        "painting", "drawing", "sculpting", "sketching", "watercolor", "oil painting",
                        "pottery", "ceramics", "knitting", "crocheting", "sewing", "embroidery", "quilting",
                        "DIY projects", "woodworking", "origami", "calligraphy", "photography", "scrapbooking",
                        "digital art", "collage", "textile art", "jewelry making", "handmade crafts",
                        "mural painting", "creative expression"
                ],
                "emotions": ["joy", "amusement", "annoyance", "pride", "relief", "curiosity", "nautral"]
            },
            "Recreation": {
                "keywords": [
                        "soccer", "basketball", "baseball", "football", "tennis", "badminton", "swimming",
                        "running", "jogging", "hiking", "climbing", "biking", "cycling", "skiing",
                        "snowboarding", "skateboarding", "martial arts", "yoga", "pilates", "aerobics",
                        "dance", "Zumba", "team sports", "individual sports", "kayaking", "canoeing",
                        "surfing", "camping", "outdoor recreation", "fishing", "golf", "gymnastics"
                ],
                "emotions": ["joy", "excitement", "pride", "frustration", "relief", "neutral"]
            },
            "Travel & adventure": {
                "keywords": [
                        "vacation", "holiday", "road trip", "sightseeing", "exploring", "backpacking",
                        "cultural travel", "city tour", "hiking trip", "nature retreat", "camping",
                        "adventure sports", "mountaineering", "skydiving", "scuba diving", "cruise",
                        "journey", "weekend getaway", "travel photography", "exploring new cultures",
                        "adventure travel", "hiking trail", "travel planning", "guided tour"
                ],
                "emotions": ["excitement", "fear", "joy", "relief", "amusement", "neutral", "curiosity"]
            }
    },
    "MEMORY & REFLECTION": {
            "Recollection": {
                "keywords": [
                        "remember", "recalling", "reminiscing", "flashback", "thinking back", "nostalgia",
                        "memories", "recollecting", "recalling moments", "past events", "recalling faces",
                        "reliving", "vivid memory", "hazy memory", "mental snapshot", "remembering details",
                        "recalling conversations", "remembering smells", "sounds", "sights", "sentimental memories",
                        "cherished moments", "treasured memories", "vivid flashback", "recalling childhood",
                        "remembering friends", "remembering family", "remembering experiences", "used to"
                ],
                "emotions": ["joy", "sadness", "love", "grief", "neutral"]
            },
            "Time passing": {
                "keywords": [
                        "getting older", "years gone by", "growing up", "time flies", "growing older",
                        "fleeting time", "passing moments", "aging", "life passing", "fleeting youth",
                        "seasons change", "life stages", "golden years", "reminiscing past", "milestones",
                        "birthdays", "anniversaries", "changing times", "reminiscing past eras",
                        "swift passage", "moving through life", "life's journey", "ephemeral", "transient",
                        "aging gracefully", "wrinkles", "memories fading", "life evolving"
                ],
                "emotions": ["sadness", "pride", "curiosity", "grief", "relief", "realization", "neutral"]
            },
            "Learning from past": {
                "keywords": [
                        "mistakes", "learned", "growth", "reflection", "wisdom", "wise", "better person",
                        "made stronger", "lessons learned", "insight", "self-improvement", "personal growth",
                        "evolving", "overcoming challenges", "hindsight", "experience", "hard-earned knowledge",
                        "resilience", "self-awareness", "mistakes as lessons", "learned from failures",
                        "self-reflection", "thinking critically", "introspection", "transformation",
                        "maturity", "gaining perspective", "lessons of life", "growth mindset", "becoming stronger"
                ],
                "emotions": ["pride", "disgust", "opimism", "realization", "neutral"]
            }
        },
        "HEALTH & BODY": {
                "Physical health concerns": {
                    "keywords": [
                            "pain", "fatigue", "sick", "illness", "treatment", "medicine", "doctor", "fever",
                            "infection", "cold", "flu", "headache", "migraine", "injury", "recovery",
                            "chronic illness", "acute illness", "surgery", "hospitalization", "check-up",
                            "symptoms", "prescription", "therapy", "inflammation", "allergy", "weakness",
                            "dizziness", "nausea", "rehabilitation", "lab tests", "blood pressure",
                            "heart rate", "health scare", "contagious", "physical strain",
                            "aches", "cramps"
                    ],
                    "emotions": ["fear", "anger", "nervousness", "sadness", "optimism", "surprise"]
                },
                "Body image": {
                    "keywords": [
                            "weight", "body", "mirror", "insecure", "diet", "appearance", "beauty standards",
                            "self-conscious", "thin", "overweight", "fit", "muscular", "slender", "curvy",
                            "petite", "tall", "short", "blemishes", "acne", "scars", "stretch marks", "cellulite",
                            "vanity", "grooming", "confidence", "self-esteem", "body positivity",
                            "body dissatisfaction", "comparison", "flawless", "natural beauty", "symmetry",
                            "makeup", "style", "self-perception", "physique", "toned", "unfit",
                            "reflection", "personal aesthetics"
                    ],
                    "emotions": ["disgust", "pride", "embarrassment", "annoyance", "neutral", "admiration"]
                },
                "Sleep/fatigue": {
                    "keywords": [
                            "insomnia", "tired", "can't sleep", "exhausted", "oversleeping", "rest", "fatigue",
                            "drowsy", "sleepy", "groggy", "restless night", "poor sleep", "napping",
                            "REM sleep", "deep sleep", "waking up tired", "disrupted sleep", "circadian rhythm",
                            "sleep deprivation", "bedtime routine", "early rising", "lethargy", "yawning",
                            "night owl", "difficulty falling asleep", "restless mind", "overworked", "burnout",
                            "sleepyhead", "fatigue from work", "energy crash"
                    ],
                    "emotions": ["annoyance", "exhaustion", "anger", "neutral", "relief", "sadness"]
                },
                "Exercise": {
                    "keywords": [
                            "workout", "gym", "run", "yoga", "active", "training", "fitness", "marathon", "jogging",
                            "cycling", "swimming", "aerobics", "pilates", "stretching", "strength training", "cardio",
                            "resistance training", "HIIT", "hiking", "sports", "team sports", "home workout",
                            "running shoes", "personal trainer", "weightlifting", "push-ups", "pull-ups",
                            "core workout", "endurance", "flexibility", "agility", "stamina", "physical activity",
                            "fitness routine", "warm-up", "cool-down", "exercise goal", "outdoor exercise"
                    ],
                    "emotions": ["optimism", "pride", "pain", "joy", "amusement", "neutral"]
                },
                "Diet & nutrition": {
                    "keywords": [
                            "breakfast", "lunch", "dinner", "snack", "brunch", "midnight snack", "home-cooked meals",
                            "takeout", "fast food", "dining out", "meal prep", "potluck", "street food", "buffet",
                            "food delivery", "casual dining", "fine dining", "family meals", "comfort food",
                            "communal meals", "grazing", "intermittent fasting", "feasting", "healthy eating",
                            "vegetarian", "vegan", "keto", "paleo", "low-carb", "high-protein", "gluten-free",
                            "dairy-free", "organic", "plant-based", "balanced diet", "indulgence", "cheat day",
                            "calories", "macronutrients", "superfoods", "supplements", "hydration", "dieting",
                            "mindful eating", "portion control", "nutrition tracking"
                    ],
                    "emotions": ["neutral", "disgust", "relief", "pride"]
                }
        },

        "TECHNOLOGY & MODERN LIFE": {
                "Social media": {
                    "keywords": [
                            "Instagram", "TikTok", "likes", "followers", "online", "validation", "scrolling", "Twitter",
                            "social media", "comments", "interactions", "posts", "shares", "reels", "stories", "selfies",
                            "hashtags", "trends", "influencer", "content creation", "engagement", "notifications", "viral",
                            "feed", "algorithm", "memes", "DM", "dm me", "tagging", "mentions", "reactions", "following",
                            "unfollowing", "digital identity", "online presence", "social comparison", "fear of missing out",
                            "social approval", "friend requests", "live streaming", "virtual community"
                    ],
                    "emotions": ["nervousness", "approval", "embarrassment", "disgust", "sadness", "pride", "neutral"]
                },
                "Digital distraction": {
                    "keywords": [
                            "notifications", "screen time", "focus", "attention span", "phone addiction", "news", "headlines",
                            "overwhelmed by info", "internet fatigue", "multitasking", "pop-ups", "digital overwhelm",
                            "information overload", "binge-watching", "scrolling endlessly", "alerts", "pings",
                            "constant connectivity", "tech burnout", "mindless scrolling", "device dependence", "distraction",
                            "checking phone repeatedly", "online rabbit hole", "endless feed", "virtual noise", "digital clutter",
                            "app notifications", "Wi-Fi", "mobile device", "streaming", "digital immersion",
                            "time sink", "ads", "advertisement"
                    ],
                    "emotions": ["annoyance", "nervousness", "anger", "amusement", "neutral"]
                }
            },

            "JUSTICE & VALUES": {
                "Fairness": {
                    "keywords": [
                            "unfair", "discrimination", "equality", "rights", "protest", "injustice", "bias", "prejudice",
                            "inequality", "oppression", "privilege", "civil rights", "unfair treatment", "systemic injustice",
                            "marginalization", "unfair rules", "social justice", "fairness", "equity", "human rights",
                            "underrepresented", "favoritism", "racism", "sexism", "ageism", "classism", "harassment", "exclusion",
                            "unequal opportunity", "standing up for justice", "accountability", "demanding fairness", "unfair advantage"
                    ],
                    "emotions": ["anger", "annoyance", "nervousness", "realization"]
                },
                "Ethics/moral dilemmas": {
                    "keywords": [
                            "right or wrong", "integrity", "conscience", "moral", "honesty", "ethical", "virtue", "values",
                            "doing the right thing", "ethical decision", "moral conflict", "responsibility", "duty",
                            "justice", "principles", "moral choice", "righteousness", "ethical behavior", "accountability",
                            "conscience-driven", "moral compass", "dilemma", "ethical standards", "principled",
                            "self-reflection on actions", "moral responsibility", "ethical judgment",
                            "fairness in actions", "personal ethics"
                    ],
                    "emotions": ["disapproval", "disgust", "pride", "nervousness", "realization", "confusion"]
                },
                "Boundaries vs exploitation": {
                    "keywords": [
                            "manipulation", "taking advantage", "toxic", "exploit", "boundaries", "betrayal", "deceit",
                            "control", "emotional manipulation", "overstepping", "abusive behavior", "unfair treatment",
                            "disrespecting limits", "coercion", "gaslighting", "violation of trust", "exploitation of power",
                            "boundary violation", "personal limits", "being used", "predatory", "unethical behavior",
                            "protecting oneself", "asserting limits", "toxic relationships", "emotional abuse",
                            "safeguarding boundaries", "unfair advantage"
                    ],
                    "emotions": ["anger", "disgust", "fear", "annoyance", "neutral", "disapproval"]
                }
        },
        "ACADEMIC": {
                "School": {
                    "keywords": [
                            "school", "class", "teacher", "classmate", "test", "notes", "high school", "freshman",
                            "senior year", "locker", "graduation", "prom", "sophomore", "classmates", "lecture",
                            "notes", "subject", "curriculum", "syllabus", "recess", "playground", "homeroom", "principal",
                            "report card", "extracurricular", "school trip", "homework", "classroom discussion",
                            "group project", "exam prep", "detention", "school event", "peer", "academic year",
                            "school spirit", "school club", "student council"
                    ],
                    "emotions": ["nervousness", "excitement", "joy", "neutral", "pride", "approval"]
                },
                "Assignments": {
                    "keywords": [
                            "homework", "assignment", "project", "due date", "studying", "procrastinating", "projects",
                            "group work", "research", "deadline", "presentation", "classwork", "essay", "term paper",
                            "lab report", "reading assignment", "book report", "coursework", "submission", "grading",
                            "peer review", "feedback", "revision", "outline", "draft", "oral presentation", "portfolio",
                            "writing task", "assignment guidelines", "project proposal", "brainstorming",
                            "problem set", "practical work"
                    ],
                    "emotions": ["nervousness", "annoyance", "pride", "curiosity", "neutral", "fear"]
                },
                "Exams": {
                    "keywords": [
                            "exam", "test", "midterm", "final", "grade", "score", "studying", "GPA", "failed", "passed",
                            "marks", "assessment", "quiz", "pop quiz", "standardized test", "oral exam", "written exam", "multiple-choice",
                            "essay exam", "practical exam", "mock exam", "revision", "exam stress", "exam prep",
                            "time management", "cheating", "retake", "curve grading", "cumulative exam", "performance evaluation",
                            "final project", "evaluation", "test anxiety"
                    ],
                    "emotions": ["fear", "nervousness", "relief", "pride", "disappointment", "sadness", "joy"]
                },
                "University/college": {
                    "keywords": [
                            "campus", "dorm", "professor", "major", "degree", "tuition", "lecture hall", "auditorium",
                            "Bachelors", "Masters", "PhD", "thesis", "dissertation", "defense", "postgraduate", "advisor",
                            "academic", "seminar", "workshop", "fraternity", "sorority", "campus life", "student union",
                            "orientation", "campus tour", "college event", "scholarship", "financial aid", "registration",
                            "credits", "elective", "compulsory course", "academic advisor", "graduate school", "faculty", "college"
                    ],
                    "emotions": ["excitement", "neutral", "overwhelm", "pride", "nervousness", "accomplishment", "fear"]
                },
                "Research": {
                    "keywords": [
                            "lab", "experiment", "paper", "data", "citation", "publication", "research", "methodology", "survey", "analysis", "dataset", "findings", "literature review", "journal", "conference", "peer review", "fieldwork", "replication study", "theoretical framework", "hypothesis", "qualitative study", "quantitative study", "case study", "research proposal", "experiment design", "lab report", "statistical analysis", "research ethics", "research assistant", "research question", "results", "data collection", "observation", "experimentation"],
                    "emotions": ["curiosity", "neutral", "nervousness", "pride", "confusion"]
                }
        }

        }
    return lexicon
