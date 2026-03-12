"""Flask application factory and main entry point."""

import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
from flask_cors import CORS

from server.routes.bank import bank_bp
from server.routes.dashboard import dashboard_bp
from server.routes.transactions import transactions_bp


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Enable CORS for iOS app
    CORS(app)

    # Register blueprints
    app.register_blueprint(bank_bp)
    app.register_blueprint(transactions_bp)
    app.register_blueprint(dashboard_bp)

    return app


app = create_app()


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    from flask import jsonify
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

