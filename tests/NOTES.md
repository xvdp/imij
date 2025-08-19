# WIP TDD 


### TODO use mock  to ensure that methods are called only onece, for instance. 
```python

from unittest.mock import MagicMock

mock_obj = MagicMock()
mock_obj.some_method.return_value = 42

result = mock_obj.some_method()
assert result == 42
mock_obj.some_method.assert_called_once()
reset_mock(self, /, *args, return_value: bool = False, **kwargs)
assert_any_call(self, /, *args, **kwargs) #assert passes if the mock has *ever* been called
assert_called(self) #  assert that the mock was called at least once
assert_called_once(self) # assert that the mock was called only once.
assert_called_once_with(self, /, *args, **kwargs) # assert that the mock was called exactly once with args
assert_called_with(self, /, *args, **kwargs) # last call  with the specified arguments.
assert_has_calls(self, calls, any_order=False) # assert the mock has been called with `mock_calls` list
attach_mock(self, mock, attribute)`method_calls` and `mock_calls` attributes of this one.
```

### todo integration tests.
