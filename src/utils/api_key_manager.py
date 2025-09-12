from typing import List


# TODO: Extend the `APIKeyManager` class so that it can manage keys for different APIs at the same time
# TODO: Need to consider whether to implement the singleton pattern (override the __new__ dander)
class APIKeyManager:
    """A class for managing and rotating API keys"""

    def __init__(self, keys: List[str]):
        self.keys = keys.copy()
        self._index = 0
        self._current_service = None

    def get_next_key(self) -> str:
        """The method returns the following API key, if any, or raises an exception"""
        if self.is_switchable:
            print(
                f"API-ключ для {self._current_service} больше недействителен. "
                "Переходим к следующему ключу."
            )
            self._index += 1
            return self.keys[self._index]
        raise Exception(
            "Закончились доступные API-ключи. "
            "Добавьте новые или дождитесь сбрасывния лимитов."
        )

    def get_current_key(self) -> str:
        """The method returns the currently used API key"""
        return self.keys[self._index]

    @property
    def is_switchable(self) -> bool:
        """The property allows to understand whether there are still unused API keys left or not"""
        if self._index + 1 < len(self.keys):
            return True
        return False