#!/usr/bin/env python3
"""Fetch games where BOTH players are titled from the chess.com public API.

Strategy: enumerate titled usernames via /pub/titled/{TITLE}, then pull each
selected player's monthly archives and keep only games where *both* usernames
are in the titled set. Output is a PGN of guaranteed titled-vs-titled games
(deduplicated), ready to feed to the C++ `extract` tool:

    data/fetch_chesscom_titled.py --max-players 40 --months 2 --out data/work/chesscom.pgn
    build/extract --no-require-titles < data/work/chesscom.pgn > data/work/chesscom.txt

(We pass --no-require-titles to extract because both-titled is already
guaranteed here by username intersection.)

Run with the project venv's Python: training/.venv/bin/python data/fetch_chesscom_titled.py
"""
import argparse
import io
import sys
import time
import urllib.request

API = "https://api.chess.com/pub"
UA = "TitledNNUE/0.1 (chess engine training; contact: hluter@icloud.com)"
TITLES = ["GM", "IM", "FM", "WGM", "WIM", "WFM", "CM", "NM", "WCM"]


def get(url, retries=4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"  ! giving up on {url}: {e}", file=sys.stderr)
                return None
            time.sleep(2 ** attempt)
    return None


def get_json(url):
    import json

    data = get(url)
    return json.loads(data) if data else None


def titled_usernames():
    users = {}
    for t in TITLES:
        j = get_json(f"{API}/titled/{t}")
        if not j:
            continue
        for u in j.get("players", []):
            users.setdefault(u.lower(), t)
        print(f"[titled] {t}: {len(j.get('players', []))} players", file=sys.stderr)
        time.sleep(0.3)
    print(f"[titled] total unique titled usernames: {len(users)}", file=sys.stderr)
    return users


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output PGN path")
    ap.add_argument("--max-players", type=int, default=40, help="titled players to crawl")
    ap.add_argument("--months", type=int, default=2, help="most recent N months per player")
    ap.add_argument("--max-games", type=int, default=0, help="stop after N kept games (0=all)")
    args = ap.parse_args()

    import chess.pgn  # python-chess (in the venv)

    titled = titled_usernames()
    if not titled:
        print("no titled users fetched (network?)", file=sys.stderr)
        return 1

    players = sorted(titled.keys())[: args.max_players]
    seen_links = set()
    kept = 0

    with open(args.out, "w", encoding="utf-8") as out:
        for pi, player in enumerate(players, 1):
            archives = get_json(f"{API}/player/{player}/games/archives")
            if not archives:
                continue
            for arch_url in archives.get("archives", [])[-args.months:]:
                pgn_bytes = get(arch_url + "/pgn")
                time.sleep(0.3)
                if not pgn_bytes:
                    continue
                stream = io.StringIO(pgn_bytes.decode("utf-8", "replace"))
                while True:
                    game = chess.pgn.read_game(stream)
                    if game is None:
                        break
                    h = game.headers
                    w = h.get("White", "").lower()
                    b = h.get("Black", "").lower()
                    if w not in titled or b not in titled:
                        continue  # require BOTH titled
                    link = h.get("Link") or h.get("Site") or ""
                    if link and link in seen_links:
                        continue
                    seen_links.add(link)
                    # Ensure title headers are present for downstream provenance.
                    h["WhiteTitle"] = titled[w]
                    h["BlackTitle"] = titled[b]
                    print(game, file=out, end="\n\n")
                    kept += 1
                    if args.max_games and kept >= args.max_games:
                        print(f"[chesscom] reached --max-games={args.max_games}", file=sys.stderr)
                        print(f"[chesscom] wrote {kept} games to {args.out}", file=sys.stderr)
                        return 0
            print(f"[chesscom] {pi}/{len(players)} {player}: kept so far {kept}", file=sys.stderr)

    print(f"[chesscom] wrote {kept} both-titled games to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
