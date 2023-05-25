from flask import Flask
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, "../templates")
static_dir = os.path.join(current_dir, "../../static")
# Crear una instancia de la aplicación Flask
app = Flask(__name__, template_folder=template_dir, static_folder= static_dir)

# Importar los módulos de las rutas y registrar las rutas en la aplicación
from src.api import routes