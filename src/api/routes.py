import json
from functools import wraps
import signal
from flask import request, Blueprint, jsonify
from src.api.ia.aprioriModule import Apriori
from src.api.ia.metricasModule import Metricas

api = Blueprint('api', __name__)

# Custom error handler decorator
def timeout_handler(signum, frame):
    raise TimeoutError('The request timed out.')

def timeout(seconds=10):
    def decorator(func):
        @wraps(func)
        def timeOutWrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Reset the alarm
        return timeOutWrapper
    return decorator

def handle_exceptions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ValueError, KeyError) as e:
            return jsonify({'error': str(e)}), 400
        except TimeoutError:
            return jsonify({'error': 'The request timed out.'}), 408
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return wrapper

@api.route('/uploado', methods=['POST'])
@timeout(10)  # Set a timeout of 10 seconds
@handle_exceptions
def uploadData():
    if 'file' not in request.files:
        raise ValueError('No file provided')

    file = request.files['file']

    if file.filename == '':
        raise ValueError('No file selected')

    # Process the uploaded file
    # ...

    return jsonify({'message': 'File uploaded successfully'})

@api.route('/apriori/data', methods=['POST'])
@timeout(10)  # Set a timeout of 10 seconds
@handle_exceptions
def aprioriData():
    file = request.json['file']
    support = float(request.json['support'])
    confidence = float(request.json['confidence'])
    lift = float(request.json['lift'])
    page = int(request.args.get('page', 1))  # Get the page number from the request
    per_page = int(request.args.get('per_page', 10))  # Get the number of items per page

    aprioriModule = Apriori(fileName=file)
    data = aprioriModule.apriori(support=support, confidence=confidence, lift=lift)
    # Calculate the start and end indices for the current page
    page_size = per_page
    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    # Get the data for the current page
    paginated_data = data.iloc[start_index:end_index].to_dict(orient='records')
    json_data = json.dumps(paginated_data)
    
    return jsonify(json_data)