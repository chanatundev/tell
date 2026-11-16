import json,urllib.request,time,os,sys
UA={"User-Agent":"TellHackathonBot/0.1 (https://example.org/tell; student-project)"}
d=json.load(open(sys.argv[1])); out=sys.argv[2]; ok=0
for o in d:
    p=os.path.join(out,o['id']+'.jpg')
    if os.path.exists(p): ok+=1; continue
    for t in range(4):
        try:
            data=urllib.request.urlopen(urllib.request.Request(o['thumb'],headers=UA),timeout=60).read()
            open(p,'wb').write(data); ok+=1; break
        except Exception as e:
            time.sleep(3*(t+1))
    time.sleep(0.3)
print("downloaded",ok,"of",len(d))
