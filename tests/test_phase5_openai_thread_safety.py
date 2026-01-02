import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class _FakeResponse:
    def __init__(self, text: str):
        self.output_text = text


class _FakeResponses:
    def __init__(self, owner_thread_id: int, call_log: list):
        self._owner_thread_id = owner_thread_id
        self._call_log = call_log

    async def create(self, **kwargs):
        tid = threading.get_ident()
        if tid != self._owner_thread_id:
            raise RuntimeError(
                "Non-thread-safe operation invoked on an event loop other than the current one"
            )
        # Enforce Responses API shape used by the RL pipeline.
        assert "model" in kwargs
        assert "instructions" in kwargs
        assert isinstance(kwargs.get("instructions"), str)
        assert "input" in kwargs
        assert isinstance(kwargs.get("input"), list)
        assert kwargs.get("max_output_tokens") is not None
        # Default in our pipeline: do not store responses server-side.
        assert kwargs.get("store") is False
        self._call_log.append((tid, kwargs.get("model")))
        await asyncio.sleep(0)
        return _FakeResponse(text="ok")


class _FakeAsyncOpenAI:
    def __init__(self, call_log: list):
        self._owner_thread_id = threading.get_ident()
        self.responses = _FakeResponses(self._owner_thread_id, call_log)

    async def close(self):
        await asyncio.sleep(0)


def _make_manager_stub():
    # Create an instance without running LLMGenerationManager.__init__ (heavy).
    from vrag_agent.generation import LLMGenerationManager

    mgr = LLMGenerationManager.__new__(LLMGenerationManager)
    mgr.verbose_frozen = 0
    mgr.config = SimpleNamespace(
        frozen_max_concurrent=8,
        frozen_max_retries=1,
        frozen_backoff_base=1.0,
        frozen_max_tokens=32,
        frozen_model="fake-model",
        frozen_total_timeout=5.0,
        frozen_async_wrapper_timeout=5.0,
    )
    return mgr


class TestPhase5OpenAIThreadSafety(unittest.TestCase):
    def test_openai_client_not_shared_across_threads(self):
        """
        Mimic Phase 6 background threads calling the OpenAI-compatible async batch wrapper.
        The AsyncOpenAI client must not be shared across threads/event loops.
        """
        import vrag_agent.generation as gen

        created_clients = []
        call_log = []

        def _fake_make_client():
            client = _FakeAsyncOpenAI(call_log=call_log)
            created_clients.append((threading.get_ident(), client))
            return client

        mgr = _make_manager_stub()
        indices = list(range(16))
        questions = ["q"] * 16
        images_list = [[] for _ in range(16)]
        errors = []
        barrier = threading.Barrier(4)

        def _worker():
            try:
                barrier.wait(timeout=5)
                out = mgr._call_frozen_generator_batch_async_wrapper(indices, questions, images_list)
                self.assertEqual(len(out), 16)
            except Exception as e:
                errors.append(e)

        with patch.object(gen, "_HAS_OPENAI_ASYNC", True), patch.object(gen, "_SELECTED_API_KEY", "fake"), patch.object(
            gen, "_SELECTED_BASE_URL", "http://fake"
        ), patch.object(gen, "_make_openai_async_client", _fake_make_client):
            threads = [threading.Thread(target=_worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        self.assertFalse(errors)
        self.assertEqual(len(created_clients), 4)
        self.assertEqual(len({tid for tid, _ in created_clients}), 4)
        self.assertTrue(call_log)

    def test_wrapper_inside_running_event_loop(self):
        """
        Mimic the wrapper branch used when an event loop is already running:
        it should offload asyncio.run(...) to a worker thread and still keep
        the client confined to that thread/loop.
        """
        import vrag_agent.generation as gen

        call_log = []

        def _fake_make_client():
            return _FakeAsyncOpenAI(call_log=call_log)

        mgr = _make_manager_stub()
        indices = list(range(8))
        questions = ["q"] * 8
        images_list = [[] for _ in range(8)]

        async def _run():
            # Called with a running loop; wrapper should use a worker thread.
            return mgr._call_frozen_generator_batch_async_wrapper(indices, questions, images_list)

        with patch.object(gen, "_HAS_OPENAI_ASYNC", True), patch.object(gen, "_SELECTED_API_KEY", "fake"), patch.object(
            gen, "_SELECTED_BASE_URL", "http://fake"
        ), patch.object(gen, "_make_openai_async_client", _fake_make_client):
            out = asyncio.run(_run())
            self.assertEqual(len(out), 8)
            self.assertTrue(call_log)


if __name__ == "__main__":
    unittest.main()
