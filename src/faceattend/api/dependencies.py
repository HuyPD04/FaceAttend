from fastapi import Request

from ..application.container import AppContainer


def get_container(request: Request) -> AppContainer:
    return request.app.state.container
