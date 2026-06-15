from contextvars import ContextVar

_current_user_id: ContextVar[str] = ContextVar("current_user_id", default="default")


def get_current_user_id() -> str:
    return _current_user_id.get()


def set_current_user_id(user_id: str) -> None:
    _current_user_id.set(user_id)
