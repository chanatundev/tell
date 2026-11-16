"""Write 100 random prompts for NEW newer-generator training fakes (v2).
Not derived from any held-out image: randomised over generic lists."""
import json, random
from pathlib import Path
random.seed(2026)
countries = [("United States","English","USD, state sales tax"),("Canada","English/French","CAD, GST/HST"),("United Kingdom","English","GBP, VAT 20%"),
 ("Germany","German","EUR, MwSt 7%/19%"),("France","French","EUR, TVA"),("Spain","Spanish","EUR, IVA"),("Italy","Italian","EUR, scontrino fiscale, IVA"),
 ("Poland","Polish","PLN, PTU A/B/C"),("Netherlands","Dutch","EUR, BTW"),("Japan","Japanese","JPY, 消費税 8%/10%"),("South Korea","Korean","KRW, 부가세"),
 ("China","Simplified Chinese","CNY"),("Taiwan","Traditional Chinese","TWD, uniform invoice"),("Thailand","Thai","THB, VAT 7%"),("Vietnam","Vietnamese","VND"),
 ("Indonesia","Indonesian","IDR, PPN"),("Malaysia","English/Malay","MYR, SST"),("Philippines","English","PHP, VAT 12%"),("India","English","INR, GST CGST/SGST"),
 ("Australia","English","AUD, GST incl."),("Brazil","Portuguese","BRL, cupom fiscal"),("Mexico","Spanish","MXN, IVA 16%"),("Nigeria","English","NGN, VAT 7.5%"),
 ("Russia","Russian","RUB, НДС"),("Ukraine","Ukrainian","UAH, ПДВ"),("Czech Republic","Czech","CZK, DPH"),("Turkey","Turkish","TRY, KDV"),("Israel","Hebrew","ILS, מע\"מ")]
shops = ["supermarket","corner convenience store","pharmacy","hardware store","cafe","fast-food restaurant","sit-down restaurant","bakery","fuel station",
 "electronics shop","bookshop","clothing store","taxi or ride receipt","hotel bill","parking machine ticket-receipt","office supplies store","garden centre","bar"]
capture = ["phone photo held in one hand, slight perspective tilt","phone photo lying on a wooden table","phone photo on a car seat","phone photo on a kitchen counter under warm light",
 "phone photo at night with flash glare","phone photo, receipt slightly crumpled and curling","phone photo, receipt folded once with a crease","phone photo next to a coffee cup",
 "flat scanner image on white","flat scanner image, slightly skewed","phone photo, dim indoor light with noise","phone photo, outdoor daylight with shadow of the phone"]
paper = ["fresh thermal paper","faded thermal paper","thermal paper with a torn top edge","long thermal strip","small narrow thermal slip","printed A5 invoice paper","carbonless handwritten-style bill pad"]
specs = []
for i in range(100):
    c, lang, cur = random.choice(countries); s = random.choice(shops)
    y, mo, d = random.randint(2019, 2026), random.randint(1, 12), random.randint(1, 28)
    h, mi = random.randint(7, 22), random.randint(0, 59)
    n_items = random.randint(2, 14)
    p = (f"Realistic {random.choice(capture)} of a {random.choice(paper)} receipt from a {s} in {c}, printed in {lang}, currency/tax style: {cur}. "
         f"Use an INVENTED shop name and invented address (no real brand or chain). Date {d:02d}/{mo:02d}/{y} {h:02d}:{mi:02d} written in the local date format. "
         f"{n_items} line items with plausible local prices, subtotal, tax lines and total that add up correctly, a payment line, and a footer message. "
         f"Show the whole receipt. No people, no real card numbers (mask as ****1234 style).")
    specs.append({"id": f"n{i:03d}", "country": c, "shop": s, "prompt": p})
out = Path(__file__).resolve().parents[1] / "data" / "raw" / "newgen_fakes" / "specs.json"
json.dump(specs, open(out, "w"), ensure_ascii=False, indent=1); print(len(specs), out)
