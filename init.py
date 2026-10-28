from flask import Flask
from flask_cors import CORS

from .config import Config
from .history import HistoryStore
from .model_service import ModelService
from .routes import bp


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    if config:
        app.config.update(config)

    CORS(app, origins=app.config["CORS_ORIGINS"])

    app.extensions["milk_model"] = ModelService(
        app.config["MODEL_PATH"], app.config["MODEL_NAME"], app.config["RANGES_PATH"]
    )
    if app.config["ENABLE_HISTORY"]:
        app.extensions["milk_history"] = HistoryStore(app.config["DB_PATH"])

    app.register_blueprint(bp)
    return app
