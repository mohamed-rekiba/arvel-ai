"""Drivers translate the stable contract to a provider's wire format.

`fake` is a first-class driver (the red-green harness and the app test double);
`openai_compatible` speaks any OpenAI-format endpoint (including a deployed
LiteLLM proxy); `litellm` wraps the LiteLLM SDK for 100+ providers.
"""
