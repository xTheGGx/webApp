from flask import Flask
import os
from src.app.router import router
from src.api.routes import api

current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, "./app/templates")
static_dir = os.path.join(current_dir, "./static")

app = Flask(__name__, template_folder=template_dir, static_folder= static_dir)

app.register_blueprint(router, url_prefix='/')
app.register_blueprint(api, url_prefix='/api/v1')