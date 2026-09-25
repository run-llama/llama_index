import mailbox
from email.message import EmailMessage

import pytest

from llama_index.readers.file import MboxReader


def _message(subject):
    message = EmailMessage()
    message["From"] = "sender@example.com"
    message["To"] = "reader@example.com"
    message["Subject"] = subject
    return message


def plain(subject, text):
    message = _message(subject)
    message.set_content(text)
    return message


def html_with_pdf(subject, html):
    # An invoice or a notification: HTML only, with a file attached.
    message = _message(subject)
    message.set_content(html, subtype="html")
    message.add_attachment(
        b"%PDF-1.4", maintype="application", subtype="pdf", filename="invoice.pdf"
    )
    return message


@pytest.fixture()
def write_mbox(tmp_path):
    def write(*messages):
        path = tmp_path / "mail.mbox"
        box = mailbox.mbox(path)
        for message in messages:
            box.add(message)
        box.close()
        return path

    return write


def contents(path):
    return [
        document.text.split("Content: ", 1)[1]
        for document in MboxReader().load_data(path)
    ]


def test_mbox_reader_reads_an_html_message_with_an_attachment(write_mbox):
    path = write_mbox(
        html_with_pdf("Invoice", "<p>Your <b>invoice</b> for September.</p>"),
        plain("Hello", "A plain message."),
    )

    assert contents(path) == ["Your invoice for September.", "A plain message."]


def test_mbox_reader_does_not_give_a_message_the_body_before_it(write_mbox):
    path = write_mbox(
        plain("Hello", "A plain message."),
        html_with_pdf("Invoice", "<p>Your invoice for September.</p>"),
    )

    assert contents(path) == ["A plain message.", "Your invoice for September."]


def test_mbox_reader_keeps_angle_brackets_in_plain_text(write_mbox):
    path = write_mbox(plain("Contact", "Write to <jane@example.com> if 3 < 4."))

    assert contents(path) == ["Write to <jane@example.com> if 3 < 4."]


def test_mbox_reader_prefers_the_plain_alternative(write_mbox):
    message = plain("Both", "The plain version.")
    message.add_alternative("<p>The HTML version.</p>", subtype="html")
    path = write_mbox(message)

    assert contents(path) == ["The plain version."]


def test_mbox_reader_keeps_a_message_without_a_text_body(write_mbox):
    message = _message("Scan")
    message.add_attachment(
        b"%PDF-1.4", maintype="application", subtype="pdf", filename="scan.pdf"
    )
    path = write_mbox(message, plain("Hello", "A plain message."))

    assert contents(path) == ["", "A plain message."]


def test_mbox_reader_reads_a_body_in_a_charset_python_does_not_know(write_mbox):
    message = (
        b"From: sender@example.com\n"
        b"Subject: Greeting\n"
        b'Content-Type: text/plain; charset="x-unknown"\n'
        b"Content-Transfer-Encoding: 8bit\n"
        b"\n" + "Grüße aus Reutlingen".encode() + b"\n"
    )
    path = write_mbox(message)

    assert contents(path) == ["Grüße aus Reutlingen"]
