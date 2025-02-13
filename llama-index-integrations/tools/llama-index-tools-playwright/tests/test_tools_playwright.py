import pytest
import json
import os
import asyncio

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.playwright import PlaywrightToolSpec

# install playwright
os.system("playwright install")

TEST_HYPERLINKS = {
    "/python/community/ambassadors",
    "/python/docs/emulation",
    "/python/docs/api-testing",
    "/python/docs/ci-intro",
    "/python/docs/docker",
    "/python/docs/codegen",
    "#running-the-example-test",
    "/python/docs/debug",
    "/python/docs/videos",
    "/python/docs/evaluating",
    "/python/docs/input",
    "/python/docs/trace-viewer-intro",
    "#__docusaurus_skipToContent_fallback",
    "/python/docs/auth",
    "/python/docs/release-notes",
    "/python/docs/pages",
    "/python/docs/trace-viewer",
    "/python/docs/intro",
    "/python/docs/webview2",
    "/python/docs/intro#installing-playwright-pytest",
    "/python/",
    "/python/docs/dialogs",
    "https://github.com/microsoft/playwright-python",
    "/python/docs/actionability",
    "https://www.linkedin.com/company/playwrightweb",
    "#installing-playwright-pytest",
    "https://pypi.org/project/pytest-playwright/",
    "/python/docs/events",
    "/python/docs/test-runners",
    "/python/docs/frames",
    "/python/docs/languages",
    "#introduction",
    "/python/docs/locators",
    "https://www.youtube.com/channel/UC46Zj8pDH5tDosqm1gd7WTg",
    "https://dev.to/playwright",
    "/python/docs/browsers",
    "/python/docs/extensibility",
    "#system-requirements",
    "/python/docs/screenshots",
    "#updating-playwright",
    "/python/docs/running-tests",
    "/python/docs/clock",
    "/docs/intro",
    "/python/community/welcome",
    "/python/docs/navigations",
    "/python/docs/codegen-intro",
    "https://stackoverflow.com/questions/tagged/playwright",
    "https://aka.ms/playwright/discord",
    "/python/docs/writing-tests",
    "/dotnet/docs/intro",
    "/python/docs/aria-snapshots",
    "/python/docs/other-locators",
    "/java/docs/intro",
    "/python/docs/chrome-extensions",
    "/python/docs/mock",
    "/python/docs/browser-contexts",
    "/python/docs/library",
    "#",
    "/python/docs/pom",
    "/python/docs/api/class-playwright",
    "/python/docs/network",
    "https://anaconda.org/Microsoft/pytest-playwright",
    "/python/docs/test-assertions",
    "/python/docs/downloads",
    "/python/docs/handles",
    "/python/docs/intro#running-the-example-test",
    "#add-example-test",
    "/python/community/learn-videos",
    "/python/community/feature-videos",
    "https://learn.microsoft.com/en-us/training/modules/build-with-playwright/",
    "https://twitter.com/playwrightweb",
    "#whats-next",
}

TEST_TEXT = """Installation | Playwright Python Skip to main content Playwright for Python Docs API Python Python Node.js Java .NET Community Search ⌘ K Getting Started Installation Writing tests Generating tests Running and debugging tests Trace viewer Setting up CI Pytest Plugin Reference Getting started - Library Release notes Guides Actions Auto-waiting API testing Assertions Authentication Browsers Chrome extensions Clock Debugging Tests Dialogs Downloads Emulation Evaluating JavaScript Events Extensibility Frames Handles Isolation Locators Mock APIs Navigations Network Other locators Pages Page object models Screenshots Snapshot testing Test generator Trace viewer Videos WebView2 Integrations Supported languages Getting Started Installation On this page Installation Introduction \u200b Playwright was created specifically to accommodate the needs of end-to-end testing. Playwright supports all modern rendering engines including Chromium, WebKit, and Firefox. Test on Windows, Linux, and macOS, locally or on CI, headless or headed with native mobile emulation. The Playwright library can be used as a general purpose browser automation tool, providing a powerful set of APIs to automate web applications, for both sync and async Python. This introduction describes the Playwright Pytest plugin, which is the recommended way to write end-to-end tests. You will learn How to install Playwright Pytest How to run the example test Installing Playwright Pytest \u200b Playwright recommends using the official Playwright Pytest plugin to write end-to-end tests. It provides context isolation, running it on multiple browser configurations out of the box. Get started by installing Playwright and running the example test to see it in action. PyPI Anaconda Install the Pytest plugin : pip install pytest-playwright Install the Pytest plugin : conda config --add channels conda-forge conda config --add channels microsoft conda install pytest-playwright Install the required browsers: playwright install Add Example Test \u200b Create a file that follows the test_ prefix convention, such as test_example.py , inside the current working directory or in a sub-directory with the code below. Make sure your test name also follows the test_ prefix convention. test_example.py import re from playwright . sync_api import Page , expect def test_has_title ( page : Page ) : page . goto ( "https://playwright.dev/" ) # Expect a title "to contain" a substring. expect ( page ) . to_have_title ( re . compile ( "Playwright" ) ) def test_get_started_link ( page : Page ) : page . goto ( "https://playwright.dev/" ) # Click the get started link. page . get_by_role ( "link" , name = "Get started" ) . click ( ) # Expects page to have a heading with the name of Installation. expect ( page . get_by_role ( "heading" , name = "Installation" ) ) . to_be_visible ( ) Running the Example Test \u200b By default tests will be run on chromium. This can be configured via the CLI options . Tests are run in headless mode meaning no browser UI will open up when running the tests. Results of the tests and test logs will be shown in the terminal. pytest Updating Playwright \u200b To update Playwright to the latest version run the following command: pip install pytest-playwright playwright -U System requirements \u200b Python 3.8 or higher. Windows 10+, Windows Server 2016+ or Windows Subsystem for Linux (WSL). macOS 13 Ventura, or later. Debian 12, Ubuntu 22.04, Ubuntu 24.04, on x86-64 and arm64 architecture. What's next \u200b Write tests using web first assertions, page fixtures and locators Run single test, multiple tests, headed mode Generate tests with Codegen See a trace of your tests Next Writing tests Introduction Installing Playwright Pytest Add Example Test Running the Example Test Updating Playwright System requirements What's next Learn Getting started Playwright Training Learn Videos Feature Videos Community Stack Overflow Discord Twitter LinkedIn More GitHub YouTube Blog Ambassadors Copyright © 2025 Microsoft"""

TEST_ELEMENTS = """[{"innerText": "Next\\nWriting tests"}]"""

TEST_SELECTOR = "#__docusaurus_skipToContent_fallback > div > div > main > div > div > div.col.docItemCol_VOVn > div > nav > a"

TEST_SELECTOR_FILL = "#__docusaurus > nav > div.navbar__inner > div.navbar__items.navbar__items--right > div.navbarSearchContainer_Bca1 > button"

TEST_VALUE = "click"


@pytest.fixture(scope="session")
def PlaywrightTool():
    browser = asyncio.get_event_loop().run_until_complete(
        PlaywrightToolSpec.create_async_playwright_browser(headless=True)
    )
    playwright_tool = PlaywrightToolSpec.from_async_browser(browser)
    yield playwright_tool
    asyncio.get_event_loop().run_until_complete(browser.close())


def test_class(PlaywrightTool):
    names_of_base_classes = [b.__name__ for b in PlaywrightToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_navigate_to(PlaywrightTool):
    asyncio.get_event_loop().run_until_complete(
        PlaywrightTool.navigate_to("https://playwright.dev/python/docs/intro")
    )
    current_page = asyncio.get_event_loop().run_until_complete(
        PlaywrightTool.get_current_page()
    )
    assert current_page == "https://playwright.dev/python/docs/intro"


def test_extract_hyperlinks(PlaywrightTool):
    hyperlinks = asyncio.get_event_loop().run_until_complete(
        PlaywrightTool.extract_hyperlinks()
    )
    assert set(json.loads(hyperlinks)) == TEST_HYPERLINKS


def test_extract_text(PlaywrightTool):
    text = asyncio.get_event_loop().run_until_complete(PlaywrightTool.extract_text())
    # different systems may have different whitespace, so we allow for some leeway
    assert abs(len(text) - len(TEST_TEXT)) < 25


def test_get_elements(PlaywrightTool):
    element = asyncio.get_event_loop().run_until_complete(
        PlaywrightTool.get_elements(selector=TEST_SELECTOR, attributes=["innerText"])
    )
    assert element == TEST_ELEMENTS
