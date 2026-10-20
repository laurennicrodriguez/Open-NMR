from nmrsuite.io._paramfile import format_param_text, parse_param_text

SAMPLE = """##TITLE= Parameter file, TopSpin 4.0.8
##JCAMPDX= 5.0
##DATATYPE= Parameter Values
##NPOINTS= 7\t$$ modification sequence number
##ORIGIN= Bruker BioSpin GmbH
$$ 2024-05-09 17:05:17.640 -0500  user@host
##$AQ_mod= 3
##$AUNM= <au_zg>
##$AUTOPOS= <17 >
##$BF1= 500.32
##$CPDPRG= (0..3)
<> <> <waltz16> <>
##$D= (0..5)
0 1 0 0
0 0.0002
##$SPNAM= (0..2)
<> <Gaus1_270.1000> <Squa100.1000>
##$SW_h= 10000
##$FQ1LIST= <>
##$O1= 3089.476
##$TE= 2.981493e+02
##END=
"""


def test_scalars_strings_and_comments() -> None:
    p = parse_param_text(SAMPLE)
    assert p["AQ_mod"] == 3
    assert p["BF1"] == 500.32
    assert p["SW_h"] == 10000
    assert p["AUNM"] == "au_zg"
    assert p["AUTOPOS"] == "17 "
    assert p["FQ1LIST"] == ""
    assert p["NPOINTS"] == 7  # trailing $$ comment removed
    assert p["TE"] == 298.1493
    assert p["ORIGIN"] == "Bruker BioSpin GmbH"


def test_arrays_span_lines() -> None:
    p = parse_param_text(SAMPLE)
    assert p["D"] == [0, 1, 0, 0, 0, 0.0002]
    assert p["CPDPRG"] == ["", "", "waltz16", ""]
    assert p["SPNAM"] == ["", "Gaus1_270.1000", "Squa100.1000"]


def test_round_trip() -> None:
    params = {"TD": 65536, "SW_h": 10000.5, "NUC1": "1H", "D": [0.0, 1.5, 2e-6], "EMPTY": ""}
    back = parse_param_text(format_param_text(params))
    for key, value in params.items():
        assert back[key] == value
