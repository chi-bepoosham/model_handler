from typing import Generator

import pytest
from flask.testing import FlaskClient

from api import app

@pytest.fixture(scope="function")
def client() -> Generator[FlaskClient, None, None]:
    """
    Provide a Flask test client for API.
    """
    with app.test_client() as c:
        yield c