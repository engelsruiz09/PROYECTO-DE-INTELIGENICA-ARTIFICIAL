###############################################################################
# ----------------------------------------------------------------------------
# Implementación "desde cero" de Naive Bayes Multinomial con:
#   - Recursividad
#   - Builder pattern

import numpy as np  # <- Añadir esta línea
import math
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