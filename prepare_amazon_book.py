# prepare_amazon_book.py
import os, csv, time

RAW_DIR = "data/amazon-book"
OUT = os.path.join(RAW_DIR, "ratings.csv")

def parse_line(line):
    line = line.strip()
    if not line: return None
    # RecBole .inter style: 'user_id:token item_id:token rating:float timestamp:float'
    if "user_id:" in line and "item_id:" in line:
        f = {}
        for tok in line.replace("\t"," ").split():
            if ":" in tok:
                k,v = tok.split(":",1); f[k]=v
        u = f.get("user_id"); i = f.get("item_id")
        r = float(f.get("rating", "5"))
        ts = int(float(f.get("timestamp", str(time.time()))))
        if r >= 4.0:
            return (u, i, 5, ts)
        return None
    # fallback: user item [rating] [timestamp]
    parts = line.replace(",", " ").split()
    if len(parts) < 2: return None
    u, i = parts[0], parts[1]
    r = 5
    if len(parts) >= 3:
        try: r = 5 if float(parts[2]) >= 4.0 else 0
        except: pass
    if r <= 0: return None
    ts = int(time.time())
    if len(parts) >= 4:
        try: ts = int(float(parts[3]))
        except: pass
    return (u, i, r, ts)

def main():
    candidates = [
        os.path.join(RAW_DIR, "amazon-book.txt"),
        os.path.join(RAW_DIR, "Amazon_Books.inter"),
        os.path.join(RAW_DIR, "Amazon_Book.inter"),
    ]
    inp = next((c for c in candidates if os.path.exists(c)), None)
    if inp is None:
        raise FileNotFoundError(
            "Missing interactions. Download amazon-book interactions (e.g., RecBole) "
            "and put a file like Amazon_Books.inter or amazon-book.txt into data/amazon-book/"
        )
    rows = []
    with open(inp, "r", errors="ignore") as f:
        for line in f:
            rec = parse_line(line)
            if rec: rows.append(rec)

    # reindex to consecutive ints
    users, items = {}, {}
    uid = iid = 0
    out_rows = []
    for (u,i,r,ts) in rows:
        if u not in users: users[u]=uid; uid+=1
        if i not in items: items[i]=iid; iid+=1
        out_rows.append((users[u], items[i], r, ts))

    os.makedirs(RAW_DIR, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["userId","itemId","rating","timestamp"]); w.writerows(out_rows)
    print(f"Amazon-Book -> wrote {len(out_rows)} rows to {OUT} (users={len(users)}, items={len(items)})")

if __name__ == "__main__":
    main()
