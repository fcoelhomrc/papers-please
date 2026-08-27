"""A minimal fake chat model for testing LangGraph agents without an API key.

None of langchain_core's built-in fakes (FakeListChatModel etc.) implement
bind_tools() - the mechanism create_agent uses to hand the model its tool
schemas - so they can't drive a tool-calling loop. This one does the minimum
to satisfy that: bind_tools() is a no-op (ignores the schemas, since we're
not testing real tool-selection reasoning here, just the graph wiring), and
_generate() replays a scripted list of AIMessage responses in order, one per
agent "turn" (counted by how many AIMessages already appear in the running
message history).
"""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeToolCallingModel(BaseChatModel):
    responses: list[AIMessage]

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        turn = sum(1 for m in messages if isinstance(m, AIMessage))
        return ChatResult(generations=[ChatGeneration(message=self.responses[turn])])

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"
