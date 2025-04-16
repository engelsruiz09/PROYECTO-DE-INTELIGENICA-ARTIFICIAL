import os
from joblib import load
from django.shortcuts import render

# BASE_DIR: sube un nivel desde este archivo (sentiment_app/views.py)
#     1) os.path.abspath(__file__)  -> /ruta/a/proyecto_ia/sentiment_app/views.py
#     2) dirname(...) -> /ruta/a/proyecto_ia/sentiment_app
#     3) dirname(...) -> /ruta/a/proyecto_ia

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'modelo_entrenado.joblib')

pipeline = load(MODEL_PATH)

def home_view(request):
    return render(request, 'sentiment_app/index.html')

def predict_view(request):
    if request.method == 'POST':
        texto = request.POST.get('texto', '')
        pred = pipeline.predict([texto])[0]

        resultado = "Positivo" if pred == 'pos' else "Negativo"
        return render(request, 'sentiment_app/index.html', {
            'resultado': resultado,
            'texto': texto
        })

    # Si no es POST, vuelve a la página principal
    return render(request, 'sentiment_app/index.html')
