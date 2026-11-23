import json
from pathlib import Path

from app.sources import djinni, dou, robota
from app.models import Job

F = Path(__file__).parent / "fixtures"


def test_djinni_feed():
    jobs = djinni.parse_feed((F / "djinni.xml").read_text())
    assert len(jobs) >= 50
    j = jobs[0]
    assert j.ext_id.isdigit() and j.url.startswith("https://djinni.co/jobs/") and j.title
    assert len(j.description) > 100


def test_djinni_jsonld():
    j = Job(source="djinni", ext_id="856232", url="u", title="t")
    djinni.apply_jsonld(j, (F / "dj_job.html").read_text())
    assert j.company == "PulseRise Technologies"
    assert j.extra["experience_years"] == 5.0


def test_dou_feed():
    jobs = dou.parse_feed((F / "dou.xml").read_text())
    assert len(jobs) == 25
    j = jobs[0]
    assert j.title == "Senior AI .NET/C# Engineer"
    assert j.company == "YozmaTech" and j.salary == "$4500–6500" and j.location == "віддалено"
    assert j.company_slug == "yozma-tech"


def test_dou_split_title_variants():
    assert dou.split_title("AI Engineer в Acme, Київ, віддалено") == ("AI Engineer", "Acme", "", "Київ, віддалено")
    assert dou.split_title("ML в Foo") == ("ML", "Foo", "", "")


def test_dou_company_size():
    assert dou.parse_company_size((F / "dou_co.html").read_text()) == 200


def test_robota():
    jobs = robota.parse_search(json.loads((F / "robota.json").read_text()))
    assert jobs and jobs[0].url.startswith("https://robota.ua/company") and jobs[0].company
