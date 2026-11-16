import json, urllib.request, urllib.parse, sys, collections
API="https://commons.wikimedia.org/w/api.php"
UA={"User-Agent":"TellHackathonBot/0.1 (https://example.org/tell; student-project)"}
def q(**p):
    p.update(format="json")
    r=urllib.request.Request(API,data=urllib.parse.urlencode(p).encode(),headers=UA)
    import time; time.sleep(0.5)
    return json.load(urllib.request.urlopen(r,timeout=30))
seen=set(); files={}; queue=[("Category:Receipts",0)]
while queue:
    cat,d=queue.pop(0)
    if cat in seen or d>2: continue
    seen.add(cat); cont={}
    while True:
        r=q(action="query",list="categorymembers",cmtitle=cat,cmlimit=500,cmtype="file|subcat",**cont)
        for m in r["query"]["categorymembers"]:
            if m["ns"]==14: queue.append((m["title"],d+1))
            else: files[m["title"]]=cat
        if "continue" in r: cont={"cmcontinue":r["continue"]["cmcontinue"]}
        else: break
print("cats",len(seen),"files",len(files),file=sys.stderr)
out=[]; titles=list(files)
for i in range(0,len(titles),50):
    r=q(action="query",titles="|".join(titles[i:i+50]),prop="imageinfo",iiprop="url|size|mime|extmetadata",iiurlwidth=1024)
    for p in r["query"]["pages"].values():
        ii=p.get("imageinfo",[{}])[0]; em=ii.get("extmetadata",{})
        g=lambda k: em.get(k,{}).get("value","")
        out.append(dict(title=p["title"],cat=files[p["title"]],mime=ii.get("mime"),w=ii.get("width"),h=ii.get("height"),
            url=ii.get("url"),thumb=ii.get("thumburl"),license=g("LicenseShortName"),date=g("DateTimeOriginal")[:40],
            artist=urllib.parse.unquote(g("Artist"))[:120],desc=g("ImageDescription")[:200],model=g("Model") if "Model" in em else ""))
json.dump(out,open(sys.argv[1],"w"),indent=1)
c=collections.Counter(o["mime"] for o in out); print(c,file=sys.stderr)
print(collections.Counter(o["license"] for o in out).most_common(12),file=sys.stderr)
