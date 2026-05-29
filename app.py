import os
import json
import requests
from flask import Flask, request, jsonify, render_template
import anthropic

app = Flask(__name__)

LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY")
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

LASTFM_BASE = "https://ws.audioscrobbler.com/2.0/"
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ---------------------------------------------------------------------------
# Last.fm helpers
# ---------------------------------------------------------------------------

def get_lastfm_track(title, artist):
    """Fetch track info from Last.fm including tags and wiki summary."""
    resp = requests.get(
        LASTFM_BASE,
        params={
            "method": "track.getInfo",
            "api_key": LASTFM_API_KEY,
            "artist": artist,
            "track": title,
            "autocorrect": 1,
            "format": "json",
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        return None

    track = data["track"]

    # Album art: Last.fm returns a list of image dicts with "size" and "#text"
    images = track.get("album", {}).get("image", [])
    image_url = next(
        (img["#text"] for img in reversed(images) if img.get("#text")),
        None,
    )

    # Top tags (up to 5)
    raw_tags = track.get("toptags", {}).get("tag", [])
    tags = [t["name"] for t in raw_tags[:5] if t.get("name")]

    # Wiki summary — strip HTML tags crudely
    wiki_summary = track.get("wiki", {}).get("summary", "")
    if wiki_summary:
        import re
        wiki_summary = re.sub(r"<[^>]+>", "", wiki_summary).strip()
        wiki_summary = wiki_summary[:400]  # keep it concise for the prompt

    listeners = int(track.get("listeners", 0) or 0)
    # Low confidence: sparse tags and no wiki means Claude has little to work from
    low_confidence = len(tags) < 2 and not wiki_summary

    return {
        "name": track.get("name", title),
        "artist": track.get("artist", {}).get("name", artist)
                  if isinstance(track.get("artist"), dict)
                  else track.get("artist", artist),
        "album": track.get("album", {}).get("title", ""),
        "image": image_url,
        "tags": tags,
        "wiki": wiki_summary,
        "listeners": listeners,
        "duration_ms": int(track.get("duration", 0) or 0),
        "low_confidence": low_confidence,
    }


# ---------------------------------------------------------------------------
# TMDB helpers
# ---------------------------------------------------------------------------

def search_tmdb_movie(movie_title):
    """Search TMDB for a movie title and return the best match's metadata."""
    # Strip the year from "Title (Year)" if Claude included it
    import re
    clean_title = re.sub(r"\s*\(\d{4}\)\s*$", "", movie_title).strip()

    resp = requests.get(
        f"{TMDB_BASE}/search/movie",
        params={
            "api_key": TMDB_API_KEY,
            "query": clean_title,
            "include_adult": False,
        },
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])

    if not results:
        return None

    movie = results[0]
    poster = f"{TMDB_IMAGE_BASE}{movie['poster_path']}" if movie.get("poster_path") else None
    backdrop = (
        f"https://image.tmdb.org/t/p/w780{movie['backdrop_path']}"
        if movie.get("backdrop_path")
        else None
    )

    return {
        "tmdb_id": movie["id"],
        "title": movie["title"],
        "year": (movie.get("release_date") or "")[:4],
        "overview": movie.get("overview", ""),
        "rating": round(movie.get("vote_average", 0), 1),
        "poster": poster,
        "backdrop": backdrop,
        "tmdb_url": f"https://www.themoviedb.org/movie/{movie['id']}",
    }


# ---------------------------------------------------------------------------
# Claude vibe matching
# ---------------------------------------------------------------------------

VIBE_RUBRIC = """
You are a vibe analyst and film curator. When given a song's metadata, match it to a real movie that shares a similar emotional vibe using the scoring method below.

SCORING METHOD
Internally score the song across all 12 dimensions (0.0–5.0 in 0.5 increments), then identify candidate films and score them the same way. Match by finding the lowest Euclidean distance across all 12 dimensions.

WRITING THE REASON
Write 2–3 sentences in plain, evocative language. Do NOT mention dimension names, scores, numbers, or rubric terminology — those are internal tools only. Instead, describe the shared feeling in sensory, experiential terms: what it feels like to sit with both works, what emotional texture they share, what kind of person or moment connects them.

IMPORTANT CONSTRAINTS
- Do NOT suggest any movie that has won the Academy Award for Best Picture.
- Prefer niche, underseen, or cult films over mainstream blockbusters.
- If the song data is sparse (few tags, no description), be conservative — lean toward grounded, literal interpretations of the title and artist name rather than abstract or extreme mood matches.

DIMENSIONS

PILLAR I: EMOTIONAL TONE
1.1 Valence          (0.0 = crushing despair → 2.5 = ambiguous → 5.0 = pure elation)
1.2 Tension          (0.0 = total stillness → 2.5 = low-grade unease → 5.0 = unbearable dread)
1.3 Warmth           (0.0 = glacial, distant → 2.5 = detached observation → 5.0 = intimate, tender)

PILLAR II: ENERGY & PACING
2.1 Intensity        (0.0 = barely a whisper → 2.5 = conversational → 5.0 = overwhelming)
2.2 Tempo feel       (0.0 = time feels frozen → 2.5 = measured, deliberate → 5.0 = relentless propulsion)
2.3 Dynamic range    (0.0 = completely flat → 2.5 = modest swells → 5.0 = violent contrast)

PILLAR III: AESTHETIC TEXTURE
3.1 Production polish (0.0 = raw, lo-fi → 2.5 = naturalistic → 5.0 = hyper-polished)
3.2 Era feel          (0.0 = deep nostalgia → 2.5 = timeless/neutral → 5.0 = hyper-modern)
3.3 Color palette feel (0.0 = muted, desaturated → 2.5 = naturalistic → 5.0 = vivid, oversaturated)

PILLAR IV: NARRATIVE & PERSPECTIVE
4.1 Perspective scale (0.0 = deeply interior → 2.5 = individual in world → 5.0 = sweeping, epic)
4.2 Arc resolution    (0.0 = unresolved, haunting → 2.5 = bittersweet → 5.0 = fully cathartic)
4.3 Thematic weight   (0.0 = pure escapism → 2.5 = grounded drama → 5.0 = existential gravity)

Your response must be ONLY a valid JSON object — no markdown, no explanation — with these exact fields:
{
  "movie": "Movie Title (Year)",
  "director": "Director Name",
  "reason": "2–3 sentences of plain, evocative language — no scores, no dimension names, just the feeling",
  "mood_tags": ["tag1", "tag2", "tag3"]
}
"""


def match_movie(track_info):
    duration_sec = track_info["duration_ms"] // 1000
    duration_fmt = f"{duration_sec // 60}m {duration_sec % 60}s" if duration_sec else "unknown"

    prompt = (
        f"Song: '{track_info['name']}' by {track_info['artist']}\n"
        f"Album: {track_info['album'] or 'unknown'}\n"
        f"Tags: {', '.join(track_info['tags']) or 'none'}\n"
        f"Listeners: {track_info['listeners']:,}\n"
        f"Duration: {duration_fmt}\n"
    )
    if track_info["wiki"]:
        prompt += f"Description: {track_info['wiki']}\n"
    if track_info["low_confidence"]:
        prompt += (
            "\nNOTE: Last.fm data for this track is sparse (few or no tags, no description). "
            "Make a conservative, grounded match based primarily on the track title and artist name. "
            "Avoid dramatic or extreme mood mappings."
        )

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=VIBE_RUBRIC,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/match", methods=["POST"])
def match():
    data = request.get_json()
    title = (data.get("title") or "").strip()
    artist = (data.get("artist") or "").strip()

    if not title or not artist:
        return jsonify({"error": "Title and artist are required."}), 400

    try:
        # Step 1: Last.fm track lookup
        track_info = get_lastfm_track(title, artist)
        if not track_info:
            return jsonify({"error": f"No Last.fm track found for '{title}' by {artist}."}), 404

        # Step 2: Claude movie match
        claude_result = match_movie(track_info)

        # Step 3: TMDB movie enrichment
        tmdb_info = search_tmdb_movie(claude_result["movie"])

        return jsonify({
            "track": {
                "name": track_info["name"],
                "artist": track_info["artist"],
                "album": track_info["album"],
                "image": track_info["image"],
                "tags": track_info["tags"],
                "listeners": track_info["listeners"],
                "low_confidence": track_info["low_confidence"],
            },
            "match": {
                **claude_result,
                "tmdb": tmdb_info,  # None if TMDB couldn't find it
            },
        })

    except requests.HTTPError as e:
        source = "Last.fm" if "audioscrobbler" in str(e.request.url) else "TMDB"
        return jsonify({"error": f"{source} API error: {e.response.status_code}"}), 502
    except json.JSONDecodeError:
        return jsonify({"error": "Claude returned an unexpected response format."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
