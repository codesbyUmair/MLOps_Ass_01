import pytest
from app import app

@pytest.fixture
def client():
    """
    Create a test client for the app
    """
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_route(client):
    """
    Test the home route
    """
    response = client.get('/')
    assert response.status_code == 200
    assert b"Welcome to the DevOps Pipeline Demo Application" in response.data

def test_health_check(client):
    """
    Test the health check route
    """
    response = client.get('/health')
    assert response.status_code == 200
    assert b"ok" in response.data