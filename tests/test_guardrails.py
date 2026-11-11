from src.guardrails import apply_guardrails, redact_pii


def test_redacts_card_ssn_email():
    text, found = redact_pii("card 4111 1111 1111 1111, ssn 123-45-6789, me@example.com")
    assert "4111" not in text and "123-45-6789" not in text and "me@example.com" not in text
    assert set(found) == {"CARD_NUMBER", "SSN", "EMAIL"}


def test_zip_and_phone_not_redacted():
    text, found = redact_pii("my zip is 98004 and number 425-555-0100")
    assert found == [] and "98004" in text


def test_frustration_triggers_escalation():
    g = apply_guardrails("This is the THIRD TIME I'm calling, absolutely unacceptable and useless!!!")
    assert g.sentiment == "negative" and g.should_escalate


def test_neutral_and_positive():
    assert apply_guardrails("What plans do you have?").sentiment == "neutral"
    assert apply_guardrails("Thank you, that was really helpful").sentiment == "positive"
