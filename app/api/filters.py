from enum import Enum
from typing import List, Optional
from pydantic import BaseModel

class Type(Enum):
    ACTUA="Actua"
    DOSSIERS="Dossiers"
    JURISDICTION="Rechtspraak"
    SYLLABI="Syllabi"
    QUESTION_ANSWER="Vraag&antwoord"
    MEDIA="Media"
    WEBTEXTS="Webtexts"
    DEPARTMENTS="Afdelingen/Kamers"
    TOOLS="Tools"
    EVENTS="Evenementen"
    COLLABORATOR="Medewerkers/Personen"
    TEMPLATE_DOCUMENTS="Modeldocumenten"

class Category(Enum):
    LEGAL="Juridisch"
    INSURANCE="Verzekeringen"
    ACCOUNTANCY="Fiscaal / Boekhouding"
    ECONOMIC="Economisch"
    TECHNICAL="Technisch"
    ENGINEERING="Bouwtchnisch"
    ENVIRONMENT="Milieu en energie"
    PROFESSIONAL="Beroepsinstituut"
    MANAGEMENT="Management"
    COMMUNICATION="Communicatie"
    LANGUAGE="Taal"
    INFORMATICS="Informatica"
    MAINTENANCE="Onderhoud & schoonmaak"
    PERSONAL_DEVELOPMENT="Persoonlijke ontwikkeling"
    REAL_ESTATE_MANAGEMENT="Vastgoed Beheer"
    REAL_ESTATE_EXPERTISE="Vastgoed expertise"
    MISCELLANEOUS="Overige / algemeen"

class Domain(Enum):
    ANTI_MONEY_LAUNDERING="Antiwitwas"
    MANAGEMENT="Beheer"
    MEDIATION="Bemiddeling"
    GOVERNMENT_FORMS="Formulieren overheid"
    GDPR="GDPR"
    COLLABORATION="Samenwerking tussen makelaars"
    REAL_ESTATE_EXPERTISE="Vastgoedexpertise"
    REAL_ESTATE_PROMOTION="Vastgoedpromotie"
    RENT="Verhuur"
    SALE="Verkoop"
    TOURIST_REAL_ESTATE="Toeristisch vastgoed"

class SortField(Enum):
    DATE="Date"
    SCORE="Score"

class SortOrder(Enum):
    ASC="asc"
    DESC="desc"

class SortFilter(BaseModel):
    field: SortField
    order: SortOrder
    
class SearchFilters(BaseModel):
    excludeTypeFilter: List[Type] = [Type.WEBTEXTS, Type.EVENTS]
    typeFilter: List[Type]
    categoryFilter: List[Category]
    domainFilter: List[Domain]
    sortingFilter: SortFilter