"""
Streamlit Community Cloud Keepalive Script
===========================================

Visits a Streamlit app with a real headless browser (not just an HTTP ping),
so it behaves like an actual user. If the app is asleep, it detects the
"Yes, get this app back up!" button and clicks it, then waits for the app
to finish loading.

Why this instead of a simple ping/uptime monitor:
Streamlit Cloud returns HTTP 200 with a static HTML shell even when the
app is asleep. A plain GET request never starts the Python app and never
establishes the WebSocket connection Streamlit needs — so it looks
successful but doesn't actually wake anything. A headless browser runs
the real page JS and can click the wake button, which does work.

Usage:
    STREAMLIT_URL="https://your-app.streamlit.app" python keepalive.py
"""

import os
import sys
import time

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

STREAMLIT_URL = os.environ.get("STREAMLIT_URL")
PAGE_LOAD_TIMEOUT_MS = 30_000
WAKE_BUTTON_TIMEOUT_MS = 8_000
POST_WAKE_WAIT_MS = 15_000

# Text Streamlit uses on the sleep screen's wake button.
WAKE_BUTTON_TEXTS = [
    "Yes, get this app back up!",
    "get this app back up",
]


def main() -> None:
    if not STREAMLIT_URL:
        print("ERROR: STREAMLIT_URL environment variable is not set.")
        sys.exit(1)

    print(f"Visiting: {STREAMLIT_URL}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            page.goto(STREAMLIT_URL, timeout=PAGE_LOAD_TIMEOUT_MS, wait_until="domcontentloaded")
        except PlaywrightTimeoutError:
            print("ERROR: Page did not load in time.")
            browser.close()
            sys.exit(1)

        # Look for the wake-up button. If it's there, the app was asleep.
        clicked = False
        for text in WAKE_BUTTON_TEXTS:
            try:
                button = page.get_by_text(text, exact=False)
                button.wait_for(state="visible", timeout=WAKE_BUTTON_TIMEOUT_MS)
                button.click()
                clicked = True
                print(f"Found and clicked wake-up button (matched: '{text}').")
                break
            except PlaywrightTimeoutError:
                continue

        if clicked:
            # Give the app time to actually spin up after the click.
            print("Waiting for app to wake up...")
            page.wait_for_timeout(POST_WAKE_WAIT_MS)
            # Re-check the button is gone (best-effort signal it woke up).
            still_sleeping = False
            for text in WAKE_BUTTON_TEXTS:
                if page.get_by_text(text, exact=False).count() > 0:
                    still_sleeping = True
                    break
            if still_sleeping:
                print("WARNING: Wake button still visible after wait. App may need more time.")
            else:
                print("SUCCESS: App appears to be awake now.")
        else:
            print("No wake-up button found. App was likely already awake.")

        browser.close()


if __name__ == "__main__":
    main()
