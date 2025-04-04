"""Test UniProt reader."""

import tempfile

import pytest

from llama_index.readers.uniprot import UniProtReader


def create_test_record() -> str:
    return """ID   002L_FRG3G              Reviewed;         320 AA.
AC   Q6GZX3;
DT   28-JUN-2011, integrated into UniProtKB/Swiss-Prot.
DT   19-JUL-2004, sequence version 1.
DT   05-FEB-2025, entry version 47.
DE   RecName: Full=Uncharacterized protein 002L;
GN   ORFNames=FV3-002L;
OS   Frog virus 3 (isolate Goorha) (FV-3).
OC   Viruses; Varidnaviria; Bamfordvirae; Nucleocytoviricota; Megaviricetes;
OC   Pimascovirales; Iridoviridae; Alphairidovirinae; Ranavirus; Frog virus 3.
OX   NCBI_TaxID=654924;
OH   NCBI_TaxID=30343; Dryophytes versicolor (chameleon treefrog).
OH   NCBI_TaxID=8404; Lithobates pipiens (Northern leopard frog) (Rana pipiens).
OH   NCBI_TaxID=45438; Lithobates sylvaticus (Wood frog) (Rana sylvatica).
OH   NCBI_TaxID=8316; Notophthalmus viridescens (Eastern newt) (Triturus viridescens).
RN   [1]
RP   NUCLEOTIDE SEQUENCE [LARGE SCALE GENOMIC DNA].
RX   PubMed=15165820; DOI=10.1016/j.virol.2004.02.019;
RA   Tan W.G., Barkman T.J., Gregory Chinchar V., Essani K.;
RT   "Comparative genomic analyses of frog virus 3, type species of the genus
RT   Ranavirus (family Iridoviridae).";
RL   Virology 323:70-84(2004).
CC   -!- SUBCELLULAR LOCATION: Host membrane {ECO:0000305}; Single-pass membrane
CC       protein {ECO:0000305}.
CC   ---------------------------------------------------------------------------
CC   Copyrighted by the UniProt Consortium, see https://www.uniprot.org/terms
CC   Distributed under the Creative Commons Attribution (CC BY 4.0) License
CC   ---------------------------------------------------------------------------
DR   EMBL; AY548484; AAT09661.1; -; Genomic_DNA.
DR   RefSeq; YP_031580.1; NC_005946.1.
DR   GeneID; 2947774; -.
DR   KEGG; vg:2947774; -.
DR   Proteomes; UP000008770; Segment.
DR   GO; GO:0033644; C:host cell membrane; IEA:UniProtKB-SubCell.
DR   GO; GO:0016020; C:membrane; IEA:UniProtKB-KW.
DR   InterPro; IPR004251; Pox_virus_G9/A16.
DR   Pfam; PF03003; Pox_G9-A16; 1.
PE   4: Predicted;
KW   Host membrane; Membrane; Reference proteome; Transmembrane;
KW   Transmembrane helix.
FT   CHAIN           1..320
FT                   /note="Uncharacterized protein 002L"
FT                   /id="PRO_0000410509"
FT   TRANSMEM        301..318
FT                   /note="Helical"
FT                   /evidence="ECO:0000255"
FT   REGION          261..294
FT                   /note="Disordered"
FT                   /evidence="ECO:0000256|SAM:MobiDB-lite"
FT   COMPBIAS        262..294
FT                   /note="Pro residues"
FT                   /evidence="ECO:0000256|SAM:MobiDB-lite"
SQ   SEQUENCE   320 AA;  34642 MW;  9E110808B6E328E0 CRC64;
     MSIIGATRLQ NDKSDTYSAG PCYAGGCSAF TPRGTCGKDW DLGEQTCASG FCTSQPLCAR
     IKKTQVCGLR YSSKGKDPLV SAEWDSRGAP YVRCTYDADL IDTQAQVDQF VSMFGESPSL
     AERYCMRGVK NTAGELVSRV SSDADPAGGW CRKWYSAHRG PDQDAALGSF CIKNPGAADC
     KCINRASDPV YQKVKTLHAY PDQCWYVPCA ADVGELKMGT QRDTPTNCPT QVCQIVFNML
     DDGSVTMDDV KNTINCDFSK YVPPPPPPKP TPPTPPTPPT PPTPPTPPTP PTPRPVHNRK
     VMFFVAGAVL VAILISTVRW
//"""


@pytest.fixture()
def test_file() -> str:
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".dat") as f:
        f.write(create_test_record())
        return f.name


def test_uniprot_reader(test_file: str) -> None:
    reader = UniProtReader()
    documents = reader.load_data(test_file)

    assert len(documents) == 1
    doc = documents[0]

    # Check text content
    # -------

    assert "Protein ID: 002L_FRG3G" in doc.text
    assert "Accession numbers: Q6GZX3" in doc.text
    assert "Description: RecName: Full=Uncharacterized protein 002L" in doc.text
    assert "Gene names: ORFNames=FV3-002L" in doc.text
    assert "Organism: Frog virus 3 (isolate Goorha) (FV-3)" in doc.text
    assert "Host membrane" in doc.text
    assert "Sequence length: 320 AA" in doc.text
    assert "Molecular weight: 34642 Da" in doc.text
    assert (
        "- SUBCELLULAR LOCATION: Host membrane {ECO:0000305}; Single-pass membrane"
        in doc.text
    )
    assert "- protein {ECO:0000305}" in doc.text

    # Check that footer comments are not in the text
    assert "Copyrighted" not in doc.text
    assert "Distributed" not in doc.text

    assert (
        "Taxonomy:\n  Viruses > Varidnaviria > Bamfordvirae > Nucleocytoviricota > Megaviricetes > Pimascovirales > Iridoviridae > Alphairidovirinae > Ranavirus > Frog virus 3"
        in doc.text
    )

    assert "Taxonomy ID: NCBI_TaxID 654924" in doc.text
    assert "EMBL: AY548484 - AAT09661.1; -; Genomic_DNA" in doc.text
    assert "RefSeq: YP_031580.1 - NC_005946.1" in doc.text

    # Citations
    # -------

    assert "Citations:" in doc.text
    assert "Reference 1:" in doc.text
    assert "Position: NUCLEOTIDE SEQUENCE [LARGE SCALE GENOMIC DNA]" in doc.text
    assert (
        "Cross-references: PubMed=15165820; DOI=10.1016/j.virol.2004.02.019" in doc.text
    )
    assert "Authors: Tan W.G., Barkman T.J., Gregory Chinchar V., Essani K." in doc.text

    assert (
        "Title: Comparative genomic analyses of frog virus 3, type species of the genus Ranavirus (family Iridoviridae)."
        in doc.text
    )

    assert "Location: Virology 323:70-84(2004)" in doc.text

    # Check metadata
    assert doc.metadata["id"] == "002L_FRG3G"


def test_uniprot_reader_minimal(test_file: str) -> None:
    reader = UniProtReader(include_fields={"id"})
    documents = reader.load_data(test_file)

    assert len(documents) == 1
    doc = documents[0]

    assert doc.text == "Protein ID: 002L_FRG3G"
    assert doc.metadata == {"id": "002L_FRG3G"}


def create_two_test_records() -> str:
    return """ID   002L_FRG3G              Reviewed;         320 AA.
AC   Q6GZX3;
DE   RecName: Full=Uncharacterized protein 002L;
//
ID   003L_FRG3G              Reviewed;         250 AA.
AC   Q6GZX4;
DE   RecName: Full=Uncharacterized protein 003L;
//"""


@pytest.fixture()
def test_file_multiple() -> str:
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".dat") as f:
        f.write(create_two_test_records())
        return f.name


def test_uniprot_reader_multiple_records(test_file_multiple: str) -> None:
    reader = UniProtReader()
    documents = reader.load_data(test_file_multiple)

    assert len(documents) == 2
    assert documents[0].metadata["id"] == "002L_FRG3G"
    assert documents[1].metadata["id"] == "003L_FRG3G"


def test_uniprot_reader_lazy(test_file: str) -> None:
    """Test lazy loading of UniProt records."""
    reader = UniProtReader()
    documents = list(reader.lazy_load_data(test_file))

    assert len(documents) == 1
    doc = documents[0]

    # Check text content
    assert "Protein ID: 002L_FRG3G" in doc.text
    assert "Accession numbers: Q6GZX3" in doc.text
    assert "Description: RecName: Full=Uncharacterized protein 002L" in doc.text
    assert doc.metadata["id"] == "002L_FRG3G"


def test_uniprot_reader_lazy_multiple(test_file_multiple: str) -> None:
    """Test lazy loading of multiple UniProt records."""
    reader = UniProtReader()
    documents = list(reader.lazy_load_data(test_file_multiple))

    assert len(documents) == 2
    assert documents[0].metadata["id"] == "002L_FRG3G"
    assert documents[1].metadata["id"] == "003L_FRG3G"


def test_uniprot_reader_lazy_minimal(test_file: str) -> None:
    """Test lazy loading with minimal fields."""
    reader = UniProtReader(include_fields={"id"})
    documents = list(reader.lazy_load_data(test_file))

    assert len(documents) == 1
    doc = documents[0]

    assert doc.text == "Protein ID: 002L_FRG3G"
    assert doc.metadata == {"id": "002L_FRG3G"}


def test_uniprot_reader_max_records(test_file_multiple: str) -> None:
    """Test limiting the number of records parsed."""
    reader = UniProtReader(max_records=1)
    documents = reader.load_data(test_file_multiple)

    assert len(documents) == 1


def test_uniprot_reader_max_records_lazy(test_file_multiple: str) -> None:
    """Test limiting the number of records parsed with lazy loading."""
    reader = UniProtReader(max_records=1)
    documents = list(reader.lazy_load_data(test_file_multiple))

    assert len(documents) == 1


def test_uniprot_reader_count_records(test_file_multiple: str) -> None:
    """Test counting the total number of records in the database."""
    reader = UniProtReader()
    count = reader.count_records(test_file_multiple)
    assert count == 2
