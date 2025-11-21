from typing import Any, Optional


class LocalEnv:
    def __init__(self):
        self.env_dict = {}

    def dict(self):
        return self.env_dict

    def __str__(self):
        return str(self.env_dict)

    def __repr__(self):
        return "LocalEnv(" + self.__str__() + ")"

    def set_by_id(self, id: str, value: Any):
        self.env_dict[id] = value

    def get_by_id(self, id: str, raise_on_none=True):
        if raise_on_none and id not in self.env_dict:
            raise ValueError(id)
        return self.env_dict.get(id)


class Name:
    def __init__(self, id):
        self.id = id


class Closure:
    def __init__(self, name: Name | None = None, executable: Optional = None,
                 value: Optional = None, local_env: LocalEnv | None = None):
        """
        A closure is a variable with a value yet to be decided. On run(), the value is calculated and stored.
        If there is an executable, then the value is decided by that executable.
        Else, the value is predefined and stored.
        :param name:
        :param executable:
        :param value:
        :param local_env:
        :param is_none:
        """
        assert executable is None or value is None, "Only one of executable or value can be None"

        self.name = name

        self.executable = executable
        self.value = value

        self._result = None
        self._ran = False
        self.local_env = local_env
        self.register_local_env(local_env)

    def register_local_env(self, local_env: LocalEnv | None):
        self.local_env = local_env

    def run(self):
        assert not self._ran, "Closures can only be run once!"
        self._ran = True
        if self.executable:
            self._result = self.executable()  # Run the executable
        else:
            self._result = self.value  # is already stored
        if self.local_env is not None and self.name is not None:  # Update result to local env
            self.local_env.set_by_id(self.name.id, self._result)
        return self._result

    def result(self):
        if not self._ran:
            return self.run()
        return self._result

    def __str__(self):
        return self.name or f"Closure({self.executable or self.value})"

    def __repr__(self):
        return self.name or f"Closure({self.executable or self.value})"


class GetClosure(Closure):
    def __init__(self, name: Name, local_env: LocalEnv | None = None):
        """
        A special closure that fetches the value of <name> from local_env.
        It is special in that it can be called multiple times.
        :param name:
        :param local_env:
        """
        Closure.__init__(self, name, local_env=local_env)

    def run(self):
        self._ran = False
        self._result = self.local_env.get_by_id(self.name.id)
        return self._result

    def result(self):
        return self.run()
