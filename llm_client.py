"""
Thin wrapper around Gemini with the protections needed when ~25 people share one key:

  * a concurrency gate so a burst of clicks is smoothed out instead of all hitting at once
  * retries with back-off on 429 / temporary 5xx errors
  * a single QuotaExceeded exception the UI can turn into a friendly message
"""
import threading
import time

MODEL_NAME = "gemini-2.5-flash-lite"
MAX_CONCURRENT_CALLS = 4

_gate = threading.BoundedSemaphore(MAX_CONCURRENT_CALLS)


class QuotaExceeded(Exception):
    """The AI service is rate-limited or temporarily unavailable."""


def generate_text(prompt, model_name=MODEL_NAME, retries=3):
    import google.generativeai as genai
    from google.api_core import exceptions as gexc

    retryable = (
        gexc.ResourceExhausted,
        gexc.ServiceUnavailable,
        gexc.InternalServerError,
        gexc.DeadlineExceeded,
    )
    delay = 2.0
    for attempt in range(retries):
        try:
            with _gate:
                response = genai.GenerativeModel(model_name).generate_content(prompt)
            try:
                return response.text
            except ValueError:
                # Response was blocked / empty (e.g. safety filter)
                return ""
        except retryable:
            if attempt == retries - 1:
                raise QuotaExceeded("AI service is busy")
            time.sleep(delay)
            delay *= 2.5
    raise QuotaExceeded("AI service is busy")
