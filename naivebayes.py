###############################################################################
# ----------------------------------------------------------------------------
# Implementación "desde cero" de Naive Bayes Multinomial con:
#   - Recursividad
#   - Builder pattern

import math
from collections import defaultdict, Counter


class RecNBBuilder:
    """
    Builder para configurar y construir la instancia de RecNaiveBayes,
    conservando el mismo interfaz que un NB (fit, predict, etc.).
    """

    def __init__(self):
        self._laplace = 1.0  # valor por defecto

    def set_laplace(self, alpha: float):
        """
        Define el factor de suavizado Laplace (alpha).
        Regresa 'self' para encadenar llamadas.
        """
        self._laplace = alpha
        return self

    def build(self):
        """
        Retorna la instancia de RecNaiveBayes con la configuración establecida.
        """
        return RecNaiveBayes(alpha=self._laplace)


class RecNaiveBayes:
    """
    Naive Bayes Multinomial 'desde cero' con recursividad,
    pero usando métodos típicos (fit, predict, predict_proba, score)
    para que tu Notebook no cambie de lógica externa.
    """

    def __init__(self, alpha=1.0):
        """
        alpha: factor de suavizado Laplace (por defecto 1.0 => Laplace normal).
        """
        self.alpha = alpha
        self.labelList = []        # Lista de clases, p.ej. ["pos","neg","neu"]
        self.priorMap = {}         # P(clase)
        self.tokenCounts = {}      # {clase: {token: frecuencia}}
        self.totalTokensByLabel = {}   # {clase: total de tokens en esa clase}
        self.globalVocab = set()   # Vocabulario global

    def fit(self, X, y):
        """
        Ajusta el modelo con datos:
        X: lista de documentos tokenizados -> [['i','love','python'], ...]
        y: lista de etiquetas -> ['pos','neg','pos', ...]
        """
        # 1. Recoger clases únicas
        self.labelList = list(set(y))
        # 2. Contar #docs totales
        total_docs = len(X)
        # 3. Contar cuántos docs por clase
        labelCounter = Counter(y)

        # 4. Calcular P(clase) y estructuras auxiliares
        for lbl in self.labelList:
            self.priorMap[lbl] = labelCounter[lbl] / total_docs
            self.tokenCounts[lbl] = defaultdict(int)
            self.totalTokensByLabel[lbl] = 0

        # 5. Recorrer cada doc y acumular conteos
        for tokens, lbl in zip(X, y):
            for token in tokens:
                self.tokenCounts[lbl][token] += 1
                self.totalTokensByLabel[lbl] += 1
                self.globalVocab.add(token)

    def predict(self, X):
        """
        Predice la etiqueta más probable para cada documento en X.
        X: lista de documentos tokenizados.
        Return: lista con la etiqueta predicha para cada doc.
        """
        predictions = []
        for tokens in X:
            best_lbl = None
            best_score = float('-inf')
            # Evaluar cada clase
            for lbl in self.labelList:
                # log P(clase)
                log_score = math.log(self.priorMap[lbl])
                # + recursividad para P(doc|clase)
                log_score += self._recursive_log_prob(tokens, lbl, 0, {})
                if log_score > best_score:
                    best_score = log_score
                    best_lbl = lbl
            predictions.append(best_lbl)
        return predictions

    def predict_proba(self, X):
        """
        Retorna, para cada documento, un dict con la probabilidad de cada clase.
        Formato: [{lbl1: p, lbl2: p, ...}, { ... }, ... ]
        """
        results = []
        for tokens in X:
            log_scores = {}
            for lbl in self.labelList:
                current = math.log(self.priorMap[lbl])
                current += self._recursive_log_prob(tokens, lbl, 0, {})
                log_scores[lbl] = current

            # Convertir log-scores a probabilidades normalizadas
            max_log = max(log_scores.values())
            sum_exp = sum(math.exp(log_scores[l] - max_log) for l in self.labelList)
            label_probs = {}
            for lbl in self.labelList:
                label_probs[lbl] = math.exp(log_scores[lbl] - max_log) / sum_exp
            results.append(label_probs)
        return results

    def score(self, X, y_true):
        """
        Calcula la exactitud de predicción en X comparado con y_true.
        """
        y_pred = self.predict(X)
        correct = sum(1 for p, t in zip(y_pred, y_true) if p == t)
        return correct / len(y_true)

    # -------------------------------------------------------------------------
    # Métodos internos (log-prob recursiva)
    # -------------------------------------------------------------------------
    def _recursive_log_prob(self, tokens, lbl, idx, memo):
        """
        Suma recursivamente la log-prob de cada token en 'tokens' para la clase 'lbl'.
        - tokens: lista de tokens del documento
        - lbl: etiqueta/clase en evaluación
        - idx: índice actual en tokens
        - memo: dict para memoización, clave=(lbl, idx)
        """
        # Caso base
        if idx >= len(tokens):
            return 0.0

        key = (lbl, idx)
        if key in memo:
            return memo[key]

        # Frec en la clase
        tk = tokens[idx]
        freq = self.tokenCounts[lbl].get(tk, 0)

        # Smoothing
        numerator = freq + self.alpha
        denominator = self.totalTokensByLabel[lbl] + self.alpha * len(self.globalVocab)
        log_val = math.log(numerator / denominator)

        # Llamada recursiva para idx+1
        log_rest = self._recursive_log_prob(tokens, lbl, idx + 1, memo)

        memo[key] = log_val + log_rest
        return memo[key]
