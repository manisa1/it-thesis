# make_data.py
# Creates a small synthetic MovieLens-like dataset: userId,itemId,rating,timestamp
import random, time, csv, os

random.seed(42)

N_USERS = 3000
N_ITEMS = 1500
DENSITY = 0.006   # ~0.6% of the matrix has positives
POS_P = int(N_USERS * N_ITEMS * DENSITY)

out_path = os.path.join("data", "ratings.csv")
os.makedirs("data", exist_ok=True)

# Create some simple popularity so we have head/tail items
item_pop = [1 + int(1000 / (1 + i//10)) for i in range(N_ITEMS)]  # head items more likely
weights = [w / sum(item_pop) for w in item_pop]

rows = []
now = int(time.time())
for _ in range(POS_P):
    u = random.randrange(N_USERS)
    # biased pick: head items show up more
    r = random.random()
    cum, i = 0.0, 0
    for j, w in enumerate(weights):
        cum += w
        if r <= cum:
            i = j
            break
    rating = 5  # implicit positive
    ts = now - random.randint(0, 60*60*24*365)  # within past year
    rows.append((u, i, rating, ts))

# de-duplicate (user,item)
rows = list({(u,i): (u,i,r,t) for (u,i,r,t) in rows}.values())

with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["userId", "itemId", "rating", "timestamp"])
    w.writerows(rows)

print(f"Wrote {len(rows)} interactions to {out_path}")
