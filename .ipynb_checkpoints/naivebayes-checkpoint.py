import math
from collections import defaultdict, Counter

class MultinomialNaiveBayes:
    """
    Implementación de Naive Bayes Multinomial para clasificación de texto
    desde cero, sin uso de librerías que resuelvan NB directamente.
    """

    def __init__(self, alpha=1.0):
        """
        alpha: parámetro de suavizado de Laplace (alpha=1 => Laplace Smoothing).
        """
        self.alpha = alpha
        self.classes_ = None          # Lista de clases (ej: [0,1] o ["neg","pos","neu"])
        self.class_priors_ = {}       # P(clase)
        self.word_counts_ = {}        # Diccionario anidado: {clase: {palabra: frec}}
        self.class_totals_ = {}       # Cantidad total de palabras por clase
        self.vocab_ = set()           # Vocabulario total

    def fit(self, X, y):
        """
        Entrena el modelo usando los documentos en X y etiquetas en y.
        X: lista de documentos; cada documento es a su vez una lista de tokens (palabras).
        y: lista de etiquetas (clases) correspondiente a cada documento.
        """
        # 1) Identificar clases únicas
        self.classes_ = list(set(y))

        # 2) Inicializar contadores
        doc_count = len(X)
        class_counts = Counter(y)

        # 3) Calcular P(clase) = (#docs de clase / #docs total)
        for c in self.classes_:
            self.class_priors_[c] = class_counts[c] / doc_count
            self.word_counts_[c] = defaultdict(int)
            self.class_totals_[c] = 0

        # 4) Contar frecuencia de cada palabra en cada clase
        for tokens, label in zip(X, y):
            for word in tokens:
                self.word_counts_[label][word] += 1
                self.class_totals_[label] += 1
                self.vocab_.add(word)

    def predict(self, X):
        """
        Predice las etiquetas para una lista de documentos.
        X: lista de documentos a clasificar (cada doc es lista de tokens).
        Return: lista con predicción de clase para cada documento.
        """
        predictions = []
        for tokens in X:
            # Calcular la log-prob de cada clase y tomar la mayor
            class_scores = {}
            for c in self.classes_:
                # Iniciamos con log P(clase)
                log_prob_c = math.log(self.class_priors_[c])

                # Sumamos la log-prob de las palabras
                for word in tokens:
                    # Frec palabra en la clase
                    count_wc = self.word_counts_[c].get(word, 0)
                    # Con suavizado:
                    numerator = count_wc + self.alpha
                    denominator = self.class_totals_[c] + self.alpha * len(self.vocab_)
                    log_prob_c += math.log(numerator / denominator)

                class_scores[c] = log_prob_c

            # Escoger la clase con mayor probabilidad logarítmica
            best_class = max(class_scores, key=class_scores.get)
            predictions.append(best_class)
        return predictions

    def predict_proba(self, X):
        """
        Retorna las probabilidades de cada clase para los documentos en X.
        X: lista de documentos (tokens).
        Return: lista de diccionarios p.ej: [{clase1: p, clase2: p, ...}, ...].
        """
        results = []
        for tokens in X:
            # Calculamos log-prob
            log_prob_dict = {}
            for c in self.classes_:
                log_prob_c = math.log(self.class_priors_[c])
                for word in tokens:
                    count_wc = self.word_counts_[c].get(word, 0)
                    numerator = count_wc + self.alpha
                    denominator = self.class_totals_[c] + self.alpha * len(self.vocab_)
                    log_prob_c += math.log(numerator / denominator)
                log_prob_dict[c] = log_prob_c
            
            # Convertir log-prob a prob normal
            # Para estabilidad numérica, restamos el log-sum-exp
            max_log = max(log_prob_dict.values())
            sum_exp = 0.0
            for c in self.classes_:
                sum_exp += math.exp(log_prob_dict[c] - max_log)
            
            probs = {}
            for c in self.classes_:
                probs[c] = math.exp(log_prob_dict[c] - max_log) / sum_exp
            
            results.append(probs)
        return results

    def score(self, X, y_true):
        """
        Calcula el accuracy del modelo en el conjunto (X, y_true).
        X: lista de documentos (tokens).
        y_true: lista de etiquetas reales.
        Return: accuracy (float)
        """
        y_pred = self.predict(X)
        correct = sum(1 for pred, true in zip(y_pred, y_true) if pred == true)
        return correct / len(y_true)

