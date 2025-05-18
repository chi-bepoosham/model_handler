from flask.testing import FlaskClient


def test_health_check(client: FlaskClient):
    """
    Test that the health check endpoint returns a 200 status code and expected JSON response.
    """
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json == {"ok": True, "data": {"status": "API is running"}, "error": None}


def test_classify_clothing_success(client: FlaskClient):
    """
    Test that clothing classification returns a 200 status code and a valid response.
    """
    response = client.post(
        "/clothing",
        json={
            "gender": "1",
            "image_url": "https://www.steelcyclewear.com/cdn/shop/products/cda05650dc8bb37cc527e48a84b8a992.jpg?v=1654104256"
        }
    )
    assert response.status_code == 200
    response_data = response.json
    assert "ok" in response_data
    assert "data" in response_data


def test_classify_clothing_invalid_input(client: FlaskClient):
    """
    Test that clothing classification returns a 400 status code for invalid input.
    """
    response = client.post(
        "/clothing",
        json={
            "gender": "invalid",
            "image_url": ""
        }
    )
    assert response.status_code == 400
    response_data = response.json
    assert "ok" in response_data
    assert "error" in response_data
    assert response_data["ok"] is False


def test_classify_bodytype_success(client: FlaskClient):
    """
    Test that body type classification returns a 200 status code and a valid response.
    """
    response = client.post(
        "/bodytype",
        json={
            "gender": "1",
            "image_url": "https://core.chibepoosham.app/storage/user/body_images/632a8d58f7413d0c7bc964984dd5f0a98bbe22ba1747510499898.jpeg"
        }
    )
    assert response.status_code == 200
    response_data = response.json
    assert "ok" in response_data
    assert "data" in response_data


def test_classify_bodytype_invalid_input(client: FlaskClient):
    """
    Test that body type classification returns a 400 status code for invalid input.
    """
    response = client.post(
        "/bodytype",
        json={
            "gender": "invalid",
            "image_url": ""
        }
    )
    assert response.status_code == 400
    response_data = response.json
    assert "ok" in response_data
    assert "error" in response_data
    assert response_data["ok"] is False
