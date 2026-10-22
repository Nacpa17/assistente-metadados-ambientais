"""Mini-corpus SINTÉTICO no formato JSON-LD lido por parse_record.

ATENÇÃO: são dados de TESTE, não metadados reais da ARM. Nomes e estrutura
imitam o repositório (para exercitar o parser e a busca), mas os DOIs usam
o sufixo fictício 10.5439/00000xx e descrições/pessoas são inventadas.
Os poucos valores do 1twrmr (período, coordenadas, site) são os mesmos do
benchmark do próprio projeto (benchmarks/gold_qa.jsonl).
"""

from __future__ import annotations

import json
from pathlib import Path

SGP_CITATION = (
    "Equipe Teste. {title}. <span id='site'>Southern Great Plains (SGP)</span>"
    " <span id='facility'>Central Facility, Lamont, OK (C1)</span>. "
    "ARM Data Center (fictício).")


def _ds(code: str, title: str, *, doi_n: int, period: str | None,
        lat: float | None, lon: float | None, facility: str | None,
        instrument: str | None, variables: list[tuple[str, str]],
        keywords: list[str], description: str | None,
        citation: str | None = None, fmt: str = "NetCDF",
        creators: list[tuple[str, str, str | None]] | None = None) -> dict:
    d: dict = {
        "@context": "https://schema.org/",
        "@type": "Dataset",
        "name": f"{title} - {code}",
        "url": f"https://doi.org/10.5439/00000{doi_n:02d}",
        "keywords": keywords,
        "variableMeasured": [{"name": n, "description": desc}
                             for n, desc in variables],
        "provider": {"name": "ARM Data Center"},
        "distribution": {"name": "ARM Data Discovery",
                         "url": f"https://adc.arm.gov/discovery/#/{code}",
                         "encodingFormat": fmt},
        "isAccessibleForFree": True,
        "datePublished": "2019-03-01",
        "dateModified": "2024-05-10",
        "includedInDataCatalog": {"name": "ARM Data Center",
                                  "url": "https://adc.arm.gov"},
    }
    if description:
        d["description"] = description
    if period:
        d["temporalCoverage"] = period
    if lat is not None:
        d["spatialCoverage"] = [{"@type": "Place", "geo": {
            "@type": "GeoCoordinates", "latitude": lat, "longitude": lon}}]
    about = []
    if facility:
        about.append({"name": f"Facility Information: {facility}",
                      "url": "https://www.arm.gov/capabilities/observatories"})
    if instrument:
        about.append({"name": f"Instrument Information: {instrument}",
                      "url": "https://www.arm.gov/capabilities/instruments"})
    if about:
        d["about"] = about
    if citation:
        d["citation"] = citation
    if creators:
        d["creator"] = [{"name": n, "affiliation": {"name": aff},
                         **({"description": role} if role else {})}
                        for n, aff, role in creators]
    return d


def synthetic_corpus() -> list[dict]:
    twr_vars = [
        ("mixing_ratio", "Water vapor mixing ratio"),
        ("temp", "Air temperature"),
        ("rh", "Relative humidity"),
        ("pres", "Atmospheric pressure"),
    ]
    return [
        _ds("1twrmr",
            "Sixty Meter Tower: mixing ratio at surface, 25-m, and 60-m, "
            "1-min average",
            doi_n=1, period="1998-04-01/..", lat=36.607322, lon=-97.487643,
            facility="Southern Great Plains (SGP)", instrument="TWR",
            variables=twr_vars,
            keywords=["Atmospheric moisture", "mixing_ratio", "ARM",
                      "Surface meteorology"],
            description="Water vapor mixing ratio at 25 m and 60 m of the "
                        "tower, with best estimates of temperature, relative "
                        "humidity and pressure. 1-min average.",
            citation=SGP_CITATION.format(title="Sixty Meter Tower 1-min"),
            creators=[("Pessoa Ficticia A", "Laboratorio Teste",
                       "Instrument Mentor")]),
        _ds("30twrmr",
            "Sixty Meter Tower: mixing ratio at surface, 25-m, and 60-m, "
            "30-min average",
            doi_n=2, period="1998-04-01/2010-12-31", lat=36.607322,
            lon=-97.487643, facility="Southern Great Plains (SGP)",
            instrument="TWR", variables=twr_vars,
            keywords=["Atmospheric moisture", "Surface meteorology"],
            description="Water vapor mixing ratio at 25 m and 60 m of the "
                        "tower. 30-min average.",
            citation=SGP_CITATION.format(title="Sixty Meter Tower 30-min")),
        _ds("kazrge", "Ka ARM Zenith Radar (KAZR), general mode",
            doi_n=3, period="2011-01-01/..", lat=71.323, lon=-156.609,
            facility="North Slope of Alaska (NSA)", instrument="KAZR",
            variables=[("reflectivity", "Radar reflectivity factor"),
                       ("mean_doppler_velocity", "Mean Doppler velocity"),
                       ("spectral_width", "Doppler spectral width"),
                       ("snr", "Signal to noise ratio")],
            keywords=["Cloud Properties", "Radar reflectivity",
                      "Radar Doppler"],
            description="Vertically pointing Ka-band cloud radar moments.",
            creators=[("Pessoa Ficticia B", "Laboratorio Teste",
                       "Instrument Mentor")]),
        _ds("mwr3c", "Microwave Radiometer, 3 channel (MWR3C)",
            doi_n=4, period="2011-06-01/..", lat=36.605, lon=-97.485,
            facility="Southern Great Plains (SGP)", instrument="MWR3C",
            variables=[("pwv", "Precipitable water vapor"),
                       ("lwp", "Liquid water path"),
                       ("tbsky23", "Sky brightness temperature 23.8 GHz")],
            keywords=["Atmospheric moisture", "Cloud Properties"],
            description="Precipitable water vapor and liquid water path "
                        "retrieved from microwave brightness temperatures."),
        _ds("aosnephdry", "Nephelometer, dry, aerosol scattering",
            doi_n=5, period="2013-10-01/..", lat=39.091, lon=-28.026,
            facility="Eastern North Atlantic (ENA)", instrument="AOS",
            variables=[("Bs_G_Dry", "Aerosol total scattering coefficient, "
                                    "green channel, dry"),
                       ("Ba_G_Dry", "Aerosol absorption coefficient, green")],
            keywords=["Aerosols", "Aerosol optical properties"],
            description="Aerosol light scattering coefficients measured by "
                        "an integrating nephelometer."),
        # registro "pobre": sem descrição, sem coordenadas, sem facility
        _ds("ldquants", "Laser disdrometer quantities",
            doi_n=6, period=None, lat=None, lon=None, facility=None,
            instrument=None,
            variables=[("precip_rate", "Precipitation rate"),
                       ("rain_rate", "Rain rate")],
            keywords=["Precipitation"], description=None),
    ]


def write_corpus(dest: str | Path) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for d in synthetic_corpus():
        code = d["name"].rsplit(" - ", 1)[-1]
        (dest / f"{code}.json").write_text(
            json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return dest
