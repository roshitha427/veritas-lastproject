import pytest

from scripts.qa_server import QAConfigurationError, qa_settings


@pytest.mark.parametrize('value', [None, 'invalid', 'sqlite:///scratch_test',
    'mysql+pymysql://user:password@127.0.0.1/veritas',
    'mysql+pymysql://user:password@127.0.0.1/not-test',
    'mysql+mysqlconnector://user:password@127.0.0.1/veritas_test'])
def test_browser_qa_refuses_non_disposable_database_without_echoing_url(value):
    with pytest.raises(QAConfigurationError) as error:
        qa_settings({'TEST_DATABASE_URL': value, 'DATABASE_URL': 'mysql+pymysql://user:private-secret@localhost/production'})
    assert 'private-secret' not in str(error.value)
    assert 'password' not in str(error.value)


def test_browser_qa_forces_test_services_and_loopback_public_url():
    config, port = qa_settings({'TEST_DATABASE_URL': 'mysql://qa:qa-secret@127.0.0.1:3307/browser_test',
                               'QA_PORT': '5010', 'GEMINI_API_KEY': 'private-provider-key'})
    assert config['DATABASE_BACKEND'] == 'mysql'
    assert config['DATABASE_URL'].startswith('mysql+pymysql://')
    assert config['TESTING'] is True
    assert config['PUBLIC_BASE_URL'] == 'http://127.0.0.1:5010'
    assert port == 5010
    assert config['EVIDENCE_PROVIDER'] == 'none'
    assert config['GEMINI_API_KEY'] is None and config['TAVILY_API_KEY'] is None
    assert config['MAIL_HOST'] is None and config['WHISPER_MODEL'] is None


def test_browser_qa_live_evidence_requires_explicit_option():
    config, _ = qa_settings({'TEST_DATABASE_URL': 'mysql+pymysql://qa:pass@localhost/browser_test'},
                            live_evidence=True)
    assert config['EVIDENCE_PROVIDER'] == 'public'


@pytest.mark.parametrize('port', ['0', '-1', '65536', 'not-a-port'])
def test_browser_qa_refuses_invalid_port(port):
    with pytest.raises(QAConfigurationError, match='QA_PORT'):
        qa_settings({'TEST_DATABASE_URL': 'mysql+pymysql://qa:pass@localhost/browser_test', 'QA_PORT': port})
