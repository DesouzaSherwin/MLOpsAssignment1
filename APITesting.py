import pytest
from app import app
import json

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

@pytest.mark.parametrize("input_data, expected_status_code", [
    ({
        'features': [
            -118.4912, 34.0205, 30, 500, 150, 3000, 800, 6.5, '<1H OCEAN'
        ]
    }, 200),
    ({
        'features': [
            'hello', 'my', 'name', 'is', 'jethalal', 'champaklal', 'gada', 'only', 'NEAR BAY'
        ]
    }, 400)
])
def test_predict(client, input_data, expected_status_code):
    response = client.post('/predict', 
                           data=json.dumps(input_data),
                           content_type='application/json')
    assert response.status_code == expected_status_code
    if expected_status_code == 200:
        response_json = response.get_json()
        assert 'prediction' in response_json
