# # server.py
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from portfolio_data import update_maps, reset_maps, NSE_MAP, QTY_MAP

# app = Flask(__name__)
# CORS(app)  # allow frontend JS requests


# @app.route("/submit-portfolio", methods=["POST"])
# def submit_portfolio():
#     """
#     Receives stock list from webpage and updates maps.
#     """
#     try:
#         data = request.get_json(force=True)

#         if not isinstance(data, list):
#             return jsonify({"status": "error", "message": "Invalid payload"}), 400

#         update_maps(data)

#         return jsonify({
#             "status": "success",
#             "NSE_MAP": NSE_MAP,
#             "QTY_MAP": QTY_MAP
#         })

#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)}), 500


# @app.route("/reset-portfolio", methods=["POST"])
# def reset_portfolio():
#     """
#     Clears maps when browser refreshes/closes.
#     """
#     reset_maps()
#     return jsonify({"status": "cleared"})


# @app.route("/get-portfolio", methods=["GET"])
# def get_portfolio():
#     """
#     Optional: view current maps (debugging).
#     """
#     return jsonify({
#         "NSE_MAP": NSE_MAP,
#         "QTY_MAP": QTY_MAP
#     })


# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, request, jsonify
from flask_cors import CORS
from app.portfolio_data import update_maps, reset_maps, NSE_MAP, QTY_MAP
from llm_model import generate_response
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/submit-portfolio", methods=["POST"])
def submit_portfolio():
    data = request.get_json(force=True)
    print("RECEIVED FROM HTML:", data)  
    update_maps(data)
    print("UPDATED NSE_MAP:", NSE_MAP)   
    response = generate_response()  
    return jsonify({
        "status": "success",
        "analysis": response
    })


@app.route("/reset-portfolio", methods=["POST"])
def reset_portfolio():
    reset_maps()
    return jsonify({"status": "cleared"})


@app.route("/get-portfolio", methods=["GET"])
def get_portfolio():
    return jsonify({
        "NSE_MAP": NSE_MAP,
        "QTY_MAP": QTY_MAP
    })



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)