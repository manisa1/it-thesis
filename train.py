# train.py
import argparse, os, random, math, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from collections import defaultdict
random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ---------------------------
# Utilities
# ---------------------------
def index_ids(df):
    uids = {u:i for i,u in enumerate(df.userId.unique().tolist())}
    iids = {i:j for j,i in enumerate(df.itemId.unique().tolist())}
    df = df.copy()
    df["u"] = df["userId"].map(uids)
    df["i"] = df["itemId"].map(iids)
    return df, uids, iids

def train_val_test_split(df, seed=42, val_frac=0.1, test_frac=0.1):
    rng = np.random.default_rng(seed)
    df = df.copy()
    users = df["u"].unique()
    user2rows = defaultdict(list)
    for idx, row in df.iterrows():
        user2rows[row["u"]].append(idx)
    val_idx, test_idx, train_idx = [], [], []
    for u, idxs in user2rows.items():
        rng.shuffle(idxs)
        n = len(idxs)
        nv = max(1, int(val_frac*n))
        nt = max(1, int(test_frac*n))
        val_idx += idxs[:nv]
        test_idx += idxs[nv:nv+nt]
        train_idx += idxs[nv+nt:]
    return df.loc[train_idx].reset_index(drop=True), df.loc[val_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)

def make_user_pos(df):
    pos = defaultdict(set)
    for u,i in zip(df["u"].values, df["i"].values):
        pos[int(u)].add(int(i))
    return pos

def recall_at_k(ranked, positives, k=20):
    ranked_k = ranked[:k]
    return len(set(ranked_k) & set(positives)) / max(1, len(positives))

def ndcg_at_k(ranked, positives, k=20):
    dcg = 0.0
    for idx, item in enumerate(ranked[:k], start=1):
        if item in positives: dcg += 1.0 / math.log2(idx+1)
    idcg = sum(1.0 / math.log2(i+1) for i in range(1, min(k, len(positives))+1))
    return dcg / idcg if idcg>0 else 0.0

# ---------------------------
# Noise (train-only)
# ---------------------------
def add_dynamic_exposure_noise(train_df, n_users, n_items, p, seed=42):
    """Add p*|train| extra 'positive' clicks sampled by popularity (exposure bias)."""
    if p <= 0: return train_df
    rng = np.random.default_rng(seed + int(time.time())%100000)
    m = len(train_df)
    add_n = int(p * m)
    # popularity from current train
    pop = np.bincount(train_df["i"].values, minlength=n_items).astype(float)
    if pop.sum() == 0: return train_df
    probs = pop / pop.sum()
    extra = []
    for _ in range(add_n):
        u = rng.integers(0, n_users)
        i = rng.choice(n_items, p=probs)
        extra.append((u,i))
    df_extra = pd.DataFrame(extra, columns=["u","i"])
    return pd.concat([train_df, df_extra], ignore_index=True)

# ---------------------------
# Model: simple MF with BPR loss
# ---------------------------
class MF_BPR(nn.Module):
    def __init__(self, n_users, n_items, k=64):
        super().__init__()
        self.U = nn.Embedding(n_users, k)
        self.I = nn.Embedding(n_items, k)
        nn.init.normal_(self.U.weight, std=0.01)
        nn.init.normal_(self.I.weight, std=0.01)

    def score(self, u, i):
        return (self.U(u) * self.I(i)).sum(-1)

    def full_scores(self):
        return self.U.weight @ self.I.weight.T  # [n_users, n_items]

# ---------------------------
# Training
# ---------------------------
def train_epoch(model, opt, user_pos, n_items, item_weights=None, device="cpu", batch_size=2048):
    model.train()
    users = list(user_pos.keys())
    rng = np.random.default_rng()
    total_loss = 0.0
    for _ in range( int( sum(len(v) for v in user_pos.values()) / batch_size ) + 1 ):
        batch_u, batch_i, batch_j = [], [], []
        for _ in range(batch_size):
            u = rng.choice(users)
            i = rng.choice(list(user_pos[u]))
            # sample a negative j
            while True:
                j = rng.integers(0, n_items)
                if j not in user_pos[u]: break
            batch_u.append(u); batch_i.append(i); batch_j.append(j)
        u = torch.tensor(batch_u, dtype=torch.long, device=device)
        i = torch.tensor(batch_i, dtype=torch.long, device=device)
        j = torch.tensor(batch_j, dtype=torch.long, device=device)
        x_ui = model.score(u,i)
        x_uj = model.score(u,j)
        # BPR loss per-sample
        loss_vec = -torch.log(torch.sigmoid(x_ui - x_uj) + 1e-12)
        if item_weights is not None:
            w = torch.tensor(item_weights[batch_i], dtype=torch.float32, device=device)
            loss = (loss_vec * w).mean()
        else:
            loss = loss_vec.mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total_loss += float(loss.item())
    return total_loss

def evaluate(model, test_df, train_pos, k=20, device="cpu"):
    model.eval()
    with torch.no_grad():
        scores = model.full_scores().detach().cpu().numpy()
    # forbid training items if you want; here we keep it simple
    user2pos_test = defaultdict(list)
    for u,i in zip(test_df["u"].values, test_df["i"].values):
        user2pos_test[int(u)].append(int(i))
    recs, ndcgs = [], []
    for u, items in user2pos_test.items():
        ranked = np.argsort(-scores[u]).tolist()
        recs.append(recall_at_k(ranked, items, k))
        ndcgs.append(ndcg_at_k(ranked, items, k))
    return float(np.mean(recs) if recs else 0.0), float(np.mean(ndcgs) if ndcgs else 0.0)

def build_pop_weights(train_df, n_items, alpha=0.5, eps=1e-6):
    pop = np.bincount(train_df["i"].values, minlength=n_items).astype(float)
    w = (pop + eps) ** (-alpha)
    w = w * (len(w) / (w.sum() + 1e-12))  # normalize to mean ~1
    return w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="data/ratings.csv")
    ap.add_argument("--model_dir", type=str, default="runs/run1")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--k_eval", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.01)
    # noise / schedule
    ap.add_argument("--noise_exposure_bias", type=float, default=0.0)
    ap.add_argument("--noise_schedule", type=str, default="none", choices=["none","ramp"])
    ap.add_argument("--noise_schedule_epochs", type=int, default=10)
    # popularity reweight (solution)
    ap.add_argument("--reweight_type", type=str, default="none", choices=["none","popularity"])
    ap.add_argument("--reweight_alpha", type=float, default=0.0)
    ap.add_argument("--reweight_ramp_epochs", type=int, default=10)

    args = ap.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    df = pd.read_csv(args.data_path)
    # binarize: rating >=4 treated as positive
    df = df[df["rating"] >= 4].copy()
    df, uidmap, iidmap = index_ids(df)
    n_users, n_items = len(uidmap), len(iidmap)

    # split
    train_df, val_df, test_df = train_val_test_split(df, seed=42, val_frac=0.1, test_frac=0.1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MF_BPR(n_users, n_items, k=args.k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    # popularity weights (fixed from initial train set)
    item_weights = None
    if args.reweight_type == "popularity" and args.reweight_alpha > 0:
        item_weights = build_pop_weights(train_df, n_items, alpha=args.reweight_alpha)

    best = {"recall": -1.0, "epoch": -1}
    for epoch in range(1, args.epochs+1):
        # schedule for dynamic noise & reweight ramp
        # noise scale goes 0->1 over noise.schedule_epochs if ramp is used
        if args.noise_schedule == "ramp" and args.noise_schedule_epochs > 0:
            noise_scale = min(1.0, epoch / max(1, args.noise_schedule_epochs))
        else:
            noise_scale = 1.0

        # build training graph with optional extra “noisy positives”
        train_df_use = train_df.copy()
        p_actual = args.noise_exposure_bias * (noise_scale if args.noise_schedule == "ramp" else 1.0)
        if p_actual > 0:
            train_df_use = add_dynamic_exposure_noise(train_df_use, n_users, n_items, p_actual)

        # build user->positives map
        user_pos = make_user_pos(train_df_use)

        # reweight ramp factor (0->1 over reweight.ramp_epochs)
        if item_weights is not None and args.reweight_ramp_epochs > 0:
            ramp = min(1.0, epoch / max(1, args.reweight_ramp_epochs))
            iw = 1.0 + (item_weights - 1.0) * ramp
        else:
            iw = item_weights

        loss = train_epoch(model, opt, user_pos, n_items, item_weights=iw, device=device)
        # quick val
        r, n = evaluate(model, val_df, make_user_pos(train_df), k=args.k_eval, device=device)
        if r > best["recall"]:
            best = {"recall": r, "epoch": epoch}
            torch.save(model.state_dict(), os.path.join(args.model_dir, "best.pt"))
        print(f"Epoch {epoch}/{args.epochs}  loss={loss:.3f}  val Recall@{args.k_eval}={r:.4f}  NDCG@{args.k_eval}={n:.4f}  (noise_p={p_actual:.3f})")

    # test with best
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "best.pt"), map_location=device))
    r, n = evaluate(model, test_df, make_user_pos(train_df), k=args.k_eval, device=device)
    out = pd.DataFrame([{"Recall@K": r, "NDCG@K": n, "K": args.k_eval}])
    out.to_csv(os.path.join(args.model_dir, "metrics.csv"), index=False)
    print("Saved metrics to", os.path.join(args.model_dir, "metrics.csv"))

if __name__ == "__main__":
    main()
