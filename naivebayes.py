###############################################################################
# ----------------------------------------------------------------------------
# Implementación "desde cero" de Naive Bayes Multinomial con:
#   - Recursividad
#   - Builder pattern

import numpy as np 
import math
import re, string 
from collections import defaultdict, Counter

# ------------------------- Función para K-Fold Estratificado -----------------

def stratified_kfold(y, k=5, seed=42):
    """
    Devuelve una lista de k arreglos NumPy con los índices de test
    para cada fold, manteniendo la proporción de clases.
    """
    # Convertir y a un array de NumPy para operaciones eficientes
    y = np.asarray(y)
    # Inicializar generador de números aleatorios con semilla para reproducibilidad
    rng = np.random.default_rng(seed)

    # 6.1.1 ) Agrupar índices por etiqueta
    # - Usamos defaultdict para crear "buckets" vacíos automáticamente
    # - Cada clave es una etiqueta única, y el valor es una lista de índices
    buckets = defaultdict(list)
    # Iterar sobre cada elemento de y (índice y etiqueta)
    for idx, label in enumerate(y):
        buckets[label].append(idx)  # Agregar índice al bucket correspondiente

    # 6.1.2 ) Inicializar los folds como listas vacías
    folds = [ [] for _ in range(k) ]  # k folds vacíos

    # 6.1.3. Repartir cada grupo de forma redonda
    for label, idxs in buckets.items():
        # Convertir índices a array de NumPy para operaciones eficientes
        idxs = np.array(idxs)
        rng.shuffle(idxs)  # Barajar los índices del grupo actual (evita sesgo)
        # Dividir en k partes (pueden ser de tamaños desiguales)
        parts = np.array_split(idxs, k)
        # Asignar cada parte a un fold específico
        for fold_id, part in enumerate(parts):
            folds[fold_id].extend(part.tolist())  # Agregar al fold correspondiente

    # 6.1.4.) Convertir cada fold a ndarray para indexación rápida
    return [np.asarray(f, dtype=int) for f in folds]


# 4.1 ) Limpieza / Preprocesamiento
#    Funcion para limpiar texto de palabras raras: quitar URL, stopwords, etc.
stop = set("""a about above after again against all am an and any are aren't as
at be because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for from
further had hadn't has hasn't have haven't having he he'd he'll he's her here
here's hers herself him himself his how how's i i'd i'll i'm i've if in into is
isn't it it's its itself let's me more most mustn't my myself no nor not of off
on once only or other ought our ours ourselves out over own same shan't she
she'd she'll she's should shouldn't so some such than that that's the their
theirs them themselves then there there's these they they'd they'll they're
they've this those through to too under until up very was wasn't we we'd we'll
we're we've were weren't what what's when when's where where's which while who
who's whom why why's with won't would wouldn't you you'd you'll you're you've
your yours yourself yourselves""".split())

# Palabras importantes para análisis de sentimientos - NO filtrar estas
sentiment_words = set([
    # Palabras positivas
    'happy', 'love', 'great', 'good', 'nice', 'best', 'better', 'awesome',
    'amazing', 'excellent', 'fantastic', 'wonderful', 'enjoy', 'thanks', 'thank',
    'beautiful', 'perfect', 'fun', 'exciting', 'excited', 'cool', 'liked',

    # Palabras negativas
    'sad', 'hate', 'bad', 'worst', 'worse', 'terrible', 'awful', 'horrible',
    'disappointed', 'upset', 'annoyed', 'angry', 'mad', 'poor', 'sorry',
    'boring', 'failed', 'fail', 'sucks', 'suck', 'disappointing', 'broken',

    # Negaciones y modificadores (muy importantes para el sentimiento)
    'not', 'no', 'never', "n't", 'cannot', 'cant', 'wont', 'very', 'really', 'too',
    'extremely', 'totally', 'absolutely', 'completely', 'definitely'
])

# Lista final de stopwords (eliminamos palabras de sentimiento de las stopwords)
final_stopwords = stop - sentiment_words

def improved_clean_text(text):
    """Función mejorada para limpiar texto preservando elementos importantes para el sentimiento"""
    # Convertir a minúsculas
    text = text.lower()

    # Patrones para reconocer elementos especiales
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    mention_pattern = re.compile(r'@\w+')
    hashtag_pattern = re.compile(r'#(\w+)')
    emoji_pattern = re.compile(r'[:;=8][\-o\*\']?[\)\]dDpP/:\}\{@\|\\]|[\)\]dDpP/:\}\{@\|\\][\-o\*\']?[:;=8]')

    # Reemplazar patrones con tokens especiales que aportan información
    text = url_pattern.sub(' URL ', text)
    text = mention_pattern.sub(' USER ', text)
    text = hashtag_pattern.sub(r' HASHTAG_\1 ', text)  # Preservamos el contenido del hashtag

    # Convertir emojis a tokens especiales
    happy_emojis = [':)', ':-)', ':D', '=)', ':]', ':}', '=]', '=}', ':-))', ':))']
    sad_emojis = [':(', ':-(', ':[', ':{', '=(', '=[', '={', ':((', ':-((',]

    for emoji in happy_emojis:
        if emoji in text:
            text = text.replace(emoji, ' HAPPY_EMOJI ')

    for emoji in sad_emojis:
        if emoji in text:
            text = text.replace(emoji, ' SAD_EMOJI ')

    # Otras conversiones de texto a tokens semánticos
    text = text.replace('!!!', ' STRONG_EMOTION ')
    text = text.replace('!!', ' EMOTION ')
    text = text.replace('!', ' EXCL ')

    text = text.replace('???', ' STRONG_QUESTION ')
    text = text.replace('??', ' QUESTION ')
    text = text.replace('?', ' QUEST ')

    # Preservar contracciones importantes (no -> n't)
    text = text.replace("n't ", " not ")

    # Eliminar puntuación (excepto algunas que ya procesamos)
    for char in string.punctuation:
        if char not in ['!', '?']:  # Estos ya los procesamos
            text = text.replace(char, ' ')

    # Normalizar espacios
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def extract_sentiment_features(text):
    """Extrae características adicionales útiles para análisis de sentimientos"""
    features = {}

    # Contar signos de exclamación e interrogación (ya convertidos en tokens)
    features['has_exclamation'] = 'EXCL' in text or 'EMOTION' in text or 'STRONG_EMOTION' in text
    features['has_question'] = 'QUEST' in text or 'QUESTION' in text or 'STRONG_QUESTION' in text

    # Detectar emojis
    features['has_happy_emoji'] = 'HAPPY_EMOJI' in text
    features['has_sad_emoji'] = 'SAD_EMOJI' in text

    # Contar palabras en mayúsculas (énfasis)
    original_words = text.split()
    uppercase_words = sum(1 for word in original_words if word.isupper() and len(word) > 1)
    features['has_uppercase'] = uppercase_words > 0

    # Detectar negaciones
    features['has_negation'] = any(neg in text.split() for neg in ['not', 'no', 'never', 'cannot'])

    return features

def improved_tokenize(text, features):
    """Tokeniza texto y genera características mejoradas"""
    # Tokenización básica
    words = text.split()

    # Filtrar stopwords pero preservar palabras de sentimiento
    filtered_words = [word for word in words if word not in final_stopwords or word in sentiment_words]

    # Tokens base
    tokens = filtered_words

    # Añadir tokens especiales basados en características
    if features['has_exclamation']:
        tokens.append('FEATURE_EXCLAMATION')

    if features['has_question']:
        tokens.append('FEATURE_QUESTION')

    if features['has_happy_emoji']:
        tokens.append('FEATURE_HAPPY_EMOJI')

    if features['has_sad_emoji']:
        tokens.append('FEATURE_SAD_EMOJI')

    if features['has_uppercase']:
        tokens.append('FEATURE_HAS_UPPERCASE')

    if features['has_negation']:
        tokens.append('FEATURE_NEGATION')

    # Generar bigramas para capturar pares importantes
    bigrams = []
    if len(filtered_words) > 1:
        for i in range(len(filtered_words) - 1):
            # Si hay una negación seguida de una palabra de sentimiento
            if filtered_words[i] in ['not', 'no', 'never'] and filtered_words[i+1] in sentiment_words:
                bigrams.append(f"NEG_{filtered_words[i]}_{filtered_words[i+1]}")
            # O simplemente un bigrama normal
            else:
                bigrams.append(f"{filtered_words[i]}_{filtered_words[i+1]}")

    # Combinar todos los tokens
    all_tokens = tokens + bigrams

    return all_tokens

def preprocess_to_tokens(raw_text:str):
    clean = improved_clean_text(raw_text)
    feats = extract_sentiment_features(clean)
    return improved_tokenize(clean, feats)          # lista de tokens 




# ------------------------- Clase RecNBBuilder  -----------------

class RecNBBuilder:
    """
    Builder para configurar y construir instancias de RecNaiveBayes.
    Permite encadenamiento de métodos para configuración fluida.
    """
    def __init__(self):
        self._laplace = 1.0  # Valor por defecto para suavizado Laplace

    def set_laplace(self, alpha: float):
        """Configura el factor de suavizado Laplace (alpha >= 0)"""
        self._laplace = alpha
        return self  # Permite encadenamiento: builder.set_laplace(1).build()

    def build(self):
        """Crea una instancia de RecNaiveBayes con la configuración actual"""
        return RecNaiveBayes(alpha=self._laplace)

class RecNaiveBayes:
    """
    Implementación de Naive Bayes Multinomial con:
    - Recursividad para cálculo de probabilidades
    - Suavizado Laplace
    - Métodos estándar (fit/predict/proba/score)
    """
    def __init__(self, alpha=1.0):
        # Parámetro de suavizado Laplace
        self.alpha = alpha
        
        # Estructuras de datos para el modelo
        self.labelList = []        # Lista de clases únicas (ej: ['pos', 'neg'])
        self.priorMap = {}         # Probabilidades a priori P(clase)
        self.tokenCounts = {}      # Conteos de tokens por clase: {clase: {token: count}}
        self.totalTokensByLabel = {}   # Total de tokens por clase (para normalización)
        self.globalVocab = set()    # Vocabulario único de todas las clases

    def fit(self, X, y):
        """
        Entrena el modelo con documentos y etiquetas.
        X: Lista de documentos (cada doc es lista de tokens)
        y: Lista de etiquetas correspondientes
        """
        # Paso 1: Identificar clases únicas y contar documentos
        self.labelList = list(set(y))  # Eliminar duplicados
        total_docs = len(X)          # Total de documentos de entrenamiento
        label_counter = Counter(y)   # Conteo de documentos por clase

        # Paso 2: Inicializar estructuras para cada clase
        for lbl in self.labelList:
            # Probabilidad a priori: P(clase) = docs_en_clase / docs_totales
            self.priorMap[lbl] = label_counter[lbl] / total_docs
            
            # Inicializar contadores para tokens de esta clase
            self.tokenCounts[lbl] = defaultdict(int)  # Token: count
            self.totalTokensByLabel[lbl] = 0          # Suma total de tokens

        # Paso 3: Procesar cada documento para acumular estadísticas
        for tokens, lbl in zip(X, y):
            for token in tokens:
                # Actualizar conteos para el token en su clase
                self.tokenCounts[lbl][token] += 1
                self.totalTokensByLabel[lbl] += 1
                self.globalVocab.add(token)  # Mantener vocabulario global

    def predict(self, X):
        """Predice la clase más probable para cada documento en X"""
        predictions = []
        for tokens in X:
            best_lbl = None
            best_score = float('-inf')  # Inicializar con valor mínimo posible
            
            # Evaluar cada clase posible
            for lbl in self.labelList:
                # Calcular log-probabilidad conjunta: log(P(clase)) + log(P(doc|clase))
                log_score = math.log(self.priorMap[lbl])  # Término a priori
                log_score += self._recursive_log_prob(tokens, lbl, 0, {})  # Término likelihood
                
                # Mantener la mejor puntuación
                if log_score > best_score:
                    best_score = log_score
                    best_lbl = lbl
            
            predictions.append(best_lbl)
        return predictions

    def predict_proba(self, X):
        """Calcula probabilidades normalizadas para cada clase"""
        results = []
        for tokens in X:
            log_scores = {}
            # Calcular log-probabilidades para todas las clases
            for lbl in self.labelList:
                log_scores[lbl] = math.log(self.priorMap[lbl]) + self._recursive_log_prob(tokens, lbl, 0, {})
            
            # Convertir log-probs a probabilidades usando softmax (estable numéricamente)
            max_log = max(log_scores.values())  # Para evitar desbordamientos
            sum_exp = sum(math.exp(score - max_log) for score in log_scores.values())
            
            # Normalizar y crear diccionario de probabilidades
            label_probs = {
                lbl: math.exp(log_scores[lbl] - max_log) / sum_exp
                for lbl in self.labelList
            }
            results.append(label_probs)
        return results

    def score(self, X, y_true):
        """Calcula precisión (accuracy) del modelo"""
        y_pred = self.predict(X)
        correct = sum(1 for pred, true in zip(y_pred, y_true) if pred == true)
        return correct / len(y_true)
    
    def metrics_report(self, X, y_true):
        """Genera reporte de métricas: Precisión, Recall, F1 por clase"""
        y_pred = self.predict(X)
        classes = self.labelList
        
        # Inicializar matriz de confusión por clase
        confusion_matrix = {c: {'tp':0, 'fp':0, 'fn':0} for c in classes}
        
        # Llenar matriz de confusión
        for true, pred in zip(y_true, y_pred):
            for c in classes:
                if pred == c and true == c:
                    confusion_matrix[c]['tp'] += 1    # Verdaderos positivos
                elif pred == c and true != c:
                    confusion_matrix[c]['fp'] += 1    # Falsos positivos
                elif true == c and pred != c:
                    confusion_matrix[c]['fn'] += 1    # Falsos negativos
        
        # Calcular métricas por clase
        report = {'per_class': {}, 'macro': {'precision':0, 'recall':0, 'f1':0}}
        for c in classes:
            tp = confusion_matrix[c]['tp']
            fp = confusion_matrix[c]['fp']
            fn = confusion_matrix[c]['fn']
            
            # Precisión: TP / (TP + FP) (evitar división por cero)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            # Recall: TP / (TP + FN)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # F1: Media armónica de precisión y recall
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Almacenar métricas
            report['per_class'][c] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            # Acumular para promedios macro
            report['macro']['precision'] += precision
            report['macro']['recall'] += recall
            report['macro']['f1'] += f1
        
        # Calcular promedios macro
        n_classes = len(classes)
        for metric in ['precision', 'recall', 'f1']:
            report['macro'][metric] /= n_classes
        
        return report

    # ------------------------- Métodos internos ------------------------------
    def _recursive_log_prob(self, tokens, lbl, idx, memo):
        """
        Calcula recursivamente la log-probabilidad de un documento dado una clase.
        Utiliza memoización para optimizar cálculos repetidos.
        
        tokens: Lista de tokens del documento
        lbl: Clase objetivo
        idx: Índice actual en la lista de tokens
        memo: Diccionario para almacenar resultados precalculados {(lbl, idx): valor}
        """
        # Caso base: Fin del documento
        if idx >= len(tokens):
            return 0.0  # Probabilidad neutra en multiplicación (log(1) = 0)
        
        # Revisar memoización para evitar cálculos repetidos
        key = (lbl, idx)
        if key in memo:
            return memo[key]
        
        # Obtener token actual y su frecuencia en la clase
        token = tokens[idx]
        freq = self.tokenCounts[lbl].get(token, 0)  # 0 si el token no existe
        
        # Aplicar suavizado Laplace
        numerator = freq + self.alpha
        denominator = self.totalTokensByLabel[lbl] + self.alpha * len(self.globalVocab)
        
        # Calcular log-probabilidad para este token
        log_prob = math.log(numerator / denominator)
        
        # Calcular recursivamente para el resto de tokens
        log_rest = self._recursive_log_prob(tokens, lbl, idx + 1, memo)
        
        # Almacenar en memo y retornar suma acumulada
        memo[key] = log_prob + log_rest
        return memo[key]
