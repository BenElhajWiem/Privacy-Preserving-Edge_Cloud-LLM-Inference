from __future__ import annotations

import random
from dataclasses import asdict, dataclass

FIRST = ["Maya", "Daniel", "Leila", "Tomasz", "Aisha", "Kenji", "Sofia", "Mateo", "Priya", "Jonas", "Amara",
         "Lucas", "Noor", "Elena", "Kwame", "Ines", "Ravi", "Hana", "Oscar", "Zara", "Felix", "Yara", "Marcus",
         "Mei", "Andre", "Leah", "Samir", "Clara", "Diego", "Nadia", "Theo", "Imani", "Viktor", "Lina", "Kofi",
         "Rosa", "Arjun", "Freya", "Emeka", "Julia"]
FEMALE = ["Maya", "Leila", "Aisha", "Sofia", "Priya", "Amara", "Noor", "Elena", "Ines", "Hana", "Zara", "Yara",
          "Mei", "Leah", "Clara", "Nadia", "Imani", "Lina", "Rosa", "Freya", "Julia"]
MALE = ["Daniel", "Tomasz", "Kenji", "Mateo", "Jonas", "Lucas", "Kwame", "Ravi", "Oscar", "Felix", "Marcus",
        "Andre", "Samir", "Diego", "Theo", "Viktor", "Kofi", "Arjun", "Emeka"]
LAST = ["Okafor", "Lindqvist", "Haddad", "Novak", "Rahman", "Tanaka", "Moreau", "Castillo", "Iyer", "Becker",
        "Mensah", "Ferreira", "Qureshi", "Petrova", "Asante", "Duarte", "Kapoor", "Sato", "Nilsson", "Abiodun",
        "Brennan", "Kovacs", "Delgado", "Zhou", "Fontaine", "Adeyemi", "Varga", "Hosseini", "Lund", "Marchetti",
        "Osei", "Romero", "Shah", "Wojcik", "Bianchi", "Nakamura", "Ekwueme", "Laine", "Morales", "Fischer"]
CO_A = ["Brightline", "Northfield", "Cobalt", "Harbor", "Silverpine", "Redwick", "Bluestem", "Ironbridge",
        "Clearwater", "Ashford", "Lumen", "Granite", "Westmere", "Oakhurst", "Kestrel", "Meridale"]
CO_B = ["Logistics", "Analytics", "Foods", "Dental Clinic", "Engineering", "Print Works", "Software",
        "Hardware Supply", "Bakery", "Insurance Brokers", "Architects", "Pharmacy", "Motors", "Textiles"]
BANK_A = ["Northfield", "Crescent", "Pinecrest", "Riverbend", "Summit", "Harborview", "Elmstead", "Stonegate"]
BANK_B = ["Credit Union", "Savings Bank", "Community Bank", "Trust"]
CITIES = ["Leeds", "Porto", "Lyon", "Gdansk", "Tampere", "Graz", "Malaga", "Utrecht", "Bergen", "Cork",
          "Brno", "Aarhus", "Bologna", "Ghent", "Tartu", "Valencia"]
INSURERS = ["Evergreen Health", "Clearpath Mutual", "Northstar Care", "Bluebell Assurance", "Meadow Health Plan"]


@dataclass
class Persona:
    first: str
    last: str
    employer: str
    bank: str
    account: str
    city: str
    phone: str
    insurer: str
    member_id: str

    @property
    def name(self) -> str:
        return f"{self.first} {self.last}"

    def to_dict(self) -> dict:
        return {**asdict(self), "name": self.name}


def make_persona(seed: int, sex: str | None = None) -> Persona:
    r = random.Random(seed)
    acct = f"{r.randint(1000, 9999)}-{r.randint(1000, 9999)}-{r.randint(1000, 9999)}"
    phone = f"+44 7{r.randint(100, 999)} {r.randint(100000, 999999)}"
    member = f"{r.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{r.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{r.randint(1000000, 9999999)}"
    first_pool = FEMALE if sex == "female" else MALE if sex == "male" else FIRST
    return Persona(r.choice(first_pool), r.choice(LAST), f"{r.choice(CO_A)} {r.choice(CO_B)}",
                   f"{r.choice(BANK_A)} {r.choice(BANK_B)}", acct, r.choice(CITIES), phone, r.choice(INSURERS), member)
