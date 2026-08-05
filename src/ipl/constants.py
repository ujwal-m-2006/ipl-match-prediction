"""Domain constants: competition registry, franchise identity, venue aliases.

IPL data is messy across sources and eras -- the same franchise appears as
"Delhi Daredevils" and "Delhi Capitals", the same ground as "M Chinnaswamy
Stadium" and "M.Chinnaswamy Stadium, Bengaluru". Every mapping needed to
reconcile that lives here so there is exactly one place to fix a name.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Official iplt20.com feed registry
# ---------------------------------------------------------------------------
# The public feed host exposes one "competition" per tournament. These IDs were
# discovered by probing the feed index; `scripts/discover_competitions.py`
# re-runs that discovery and can append newly published seasons.
IPL_COMPETITIONS: dict[int, int] = {
    # season year : competition id
    2019: 8,
    2020: 24,
    2021: 33,
    2022: 60,
    2023: 107,
    2024: 148,
    2025: 203,
    2026: 284,
}

# Seasons the official feed does NOT publish. These are back-filled from
# Cricsheet, which carries ball-by-ball data for every IPL season since 2008.
CRICSHEET_ONLY_SEASONS: tuple[int, ...] = tuple(range(2008, 2019))

FEED_BASE_URL = "https://ipl-stats-sports-mechanic.s3.ap-south-1.amazonaws.com/ipl/feeds"
CRICSHEET_JSON_URL = "https://cricsheet.org/downloads/ipl_json.zip"
IPL_WEBSITE = "https://www.iplt20.com/"

# Data provenance tags stored on every row so the UI can show where a fact came from.
SOURCE_OFFICIAL = "iplt20"
SOURCE_CRICSHEET = "cricsheet"


# ---------------------------------------------------------------------------
# Franchise identity
# ---------------------------------------------------------------------------
# Canonical name -> short code. The canonical name is the franchise's *current*
# branding; historical names are folded into it by TEAM_ALIASES below so that,
# e.g., 2013 Delhi Daredevils results contribute to Delhi Capitals' history.
TEAM_CODES: dict[str, str] = {
    "Chennai Super Kings": "CSK",
    "Mumbai Indians": "MI",
    "Royal Challengers Bengaluru": "RCB",
    "Kolkata Knight Riders": "KKR",
    "Sunrisers Hyderabad": "SRH",
    "Delhi Capitals": "DC",
    "Punjab Kings": "PBKS",
    "Rajasthan Royals": "RR",
    "Gujarat Titans": "GT",
    "Lucknow Super Giants": "LSG",
    # Defunct franchises - kept distinct because they were separate entities.
    "Deccan Chargers": "DCH",
    "Pune Warriors": "PW",
    "Kochi Tuskers Kerala": "KTK",
    "Gujarat Lions": "GL",
    "Rising Pune Supergiant": "RPS",
}

# Every spelling ever observed -> canonical name. Keys are lower-cased and
# stripped before lookup by `normalize.canonical_team`.
TEAM_ALIASES: dict[str, str] = {
    # Royal Challengers - rebranded Bangalore -> Bengaluru in 2024.
    "royal challengers bangalore": "Royal Challengers Bengaluru",
    "royal challengers bengaluru": "Royal Challengers Bengaluru",
    "rcb": "Royal Challengers Bengaluru",
    # Delhi - Daredevils -> Capitals in 2019.
    "delhi daredevils": "Delhi Capitals",
    "delhi capitals": "Delhi Capitals",
    "dd": "Delhi Capitals",
    "dc": "Delhi Capitals",
    # Punjab - Kings XI Punjab -> Punjab Kings in 2021.
    "kings xi punjab": "Punjab Kings",
    "kings eleven punjab": "Punjab Kings",
    "punjab kings": "Punjab Kings",
    "kxip": "Punjab Kings",
    "pbks": "Punjab Kings",
    # Rising Pune Supergiant(s) - the franchise dropped the "s" after 2016.
    "rising pune supergiants": "Rising Pune Supergiant",
    "rising pune supergiant": "Rising Pune Supergiant",
    "rps": "Rising Pune Supergiant",
    # Pune Warriors India.
    "pune warriors": "Pune Warriors",
    "pune warriors india": "Pune Warriors",
    "pw": "Pune Warriors",
    # Straightforward franchises (aliases mostly cover codes + punctuation).
    "chennai super kings": "Chennai Super Kings",
    "csk": "Chennai Super Kings",
    "mumbai indians": "Mumbai Indians",
    "mi": "Mumbai Indians",
    "kolkata knight riders": "Kolkata Knight Riders",
    "kkr": "Kolkata Knight Riders",
    "sunrisers hyderabad": "Sunrisers Hyderabad",
    "srh": "Sunrisers Hyderabad",
    "rajasthan royals": "Rajasthan Royals",
    "rr": "Rajasthan Royals",
    "gujarat titans": "Gujarat Titans",
    "gt": "Gujarat Titans",
    "lucknow super giants": "Lucknow Super Giants",
    "lsg": "Lucknow Super Giants",
    "deccan chargers": "Deccan Chargers",
    "dch": "Deccan Chargers",
    "kochi tuskers kerala": "Kochi Tuskers Kerala",
    "ktk": "Kochi Tuskers Kerala",
    "gujarat lions": "Gujarat Lions",
    "gl": "Gujarat Lions",
}

# Franchises fielding a side in the most recent season - used to default the
# dashboard's team pickers and to seed the playoff simulator.
ACTIVE_TEAMS: tuple[str, ...] = (
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
)

# Brand colours, used consistently across every dashboard chart.
TEAM_COLORS: dict[str, str] = {
    "Chennai Super Kings": "#F9CD05",
    "Mumbai Indians": "#045093",
    "Royal Challengers Bengaluru": "#D1171B",
    "Kolkata Knight Riders": "#3A225D",
    "Sunrisers Hyderabad": "#F26522",
    "Delhi Capitals": "#17449B",
    "Punjab Kings": "#DD1F2D",
    "Rajasthan Royals": "#E93A90",
    "Gujarat Titans": "#1B2133",
    "Lucknow Super Giants": "#0057E2",
    "Deccan Chargers": "#8B5A2B",
    "Pune Warriors": "#2E8B9A",
    "Kochi Tuskers Kerala": "#6A0DAD",
    "Gujarat Lions": "#E04F16",
    "Rising Pune Supergiant": "#B8004A",
}

DEFAULT_TEAM_COLOR = "#6B7280"


# ---------------------------------------------------------------------------
# Venues
# ---------------------------------------------------------------------------
# Ground names arrive with inconsistent punctuation, sponsor prefixes and
# trailing city names. Canonical form is "<Stadium Name>" with the city stored
# separately on the venue row.
VENUE_ALIASES: dict[str, str] = {
    "m chinnaswamy stadium": "M Chinnaswamy Stadium",
    "m.chinnaswamy stadium": "M Chinnaswamy Stadium",
    "ma chidambaram stadium": "MA Chidambaram Stadium",
    "m.a. chidambaram stadium": "MA Chidambaram Stadium",
    "ma chidambaram stadium, chepauk": "MA Chidambaram Stadium",
    "ma chidambaram stadium, chepauk, chennai": "MA Chidambaram Stadium",
    "punjab cricket association stadium": "Punjab Cricket Association Stadium",
    "punjab cricket association is bindra stadium": (
        "Punjab Cricket Association IS Bindra Stadium"
    ),
    "punjab cricket association is bindra stadium, mohali": (
        "Punjab Cricket Association IS Bindra Stadium"
    ),
    "punjab cricket association is bindra stadium, mohali, chandigarh": (
        "Punjab Cricket Association IS Bindra Stadium"
    ),
    "feroz shah kotla": "Arun Jaitley Stadium",
    "arun jaitley stadium": "Arun Jaitley Stadium",
    "arun jaitley stadium, delhi": "Arun Jaitley Stadium",
    "wankhede stadium": "Wankhede Stadium",
    "wankhede stadium, mumbai": "Wankhede Stadium",
    "eden gardens": "Eden Gardens",
    "eden gardens, kolkata": "Eden Gardens",
    "sawai mansingh stadium": "Sawai Mansingh Stadium",
    "sawai mansingh stadium, jaipur": "Sawai Mansingh Stadium",
    "rajiv gandhi international stadium": "Rajiv Gandhi International Stadium",
    "rajiv gandhi international stadium, uppal": "Rajiv Gandhi International Stadium",
    "rajiv gandhi international stadium, uppal, hyderabad": (
        "Rajiv Gandhi International Stadium"
    ),
    "narendra modi stadium": "Narendra Modi Stadium",
    "narendra modi stadium, ahmedabad": "Narendra Modi Stadium",
    "sardar patel stadium, motera": "Narendra Modi Stadium",
    "bharat ratna shri atal bihari vajpayee ekana cricket stadium": (
        "Ekana Cricket Stadium"
    ),
    "bharat ratna shri atal bihari vajpayee ekana cricket stadium, lucknow": (
        "Ekana Cricket Stadium"
    ),
    "ekana cricket stadium": "Ekana Cricket Stadium",
    "maharashtra cricket association stadium": "Maharashtra Cricket Association Stadium",
    "maharashtra cricket association stadium, pune": (
        "Maharashtra Cricket Association Stadium"
    ),
    "dr dy patil sports academy": "Dr DY Patil Sports Academy",
    "dr. dy patil sports academy": "Dr DY Patil Sports Academy",
    "dr dy patil sports academy, mumbai": "Dr DY Patil Sports Academy",
    "himachal pradesh cricket association stadium": (
        "Himachal Pradesh Cricket Association Stadium"
    ),
    "barsapara cricket stadium": "Barsapara Cricket Stadium",
    "aca-vdca cricket stadium": "ACA-VDCA Cricket Stadium",
    "dr. y.s. rajasekhara reddy aca-vdca cricket stadium": "ACA-VDCA Cricket Stadium",
}

# Stadium -> city, for grounds whose feed rows omit the city.
VENUE_CITIES: dict[str, str] = {
    "M Chinnaswamy Stadium": "Bengaluru",
    "MA Chidambaram Stadium": "Chennai",
    "Wankhede Stadium": "Mumbai",
    "Eden Gardens": "Kolkata",
    "Arun Jaitley Stadium": "Delhi",
    "Sawai Mansingh Stadium": "Jaipur",
    "Rajiv Gandhi International Stadium": "Hyderabad",
    "Narendra Modi Stadium": "Ahmedabad",
    "Ekana Cricket Stadium": "Lucknow",
    "Punjab Cricket Association IS Bindra Stadium": "Mohali",
    "Punjab Cricket Association Stadium": "Mohali",
    "Maharashtra Cricket Association Stadium": "Pune",
    "Dr DY Patil Sports Academy": "Mumbai",
    "Barsapara Cricket Stadium": "Guwahati",
    "ACA-VDCA Cricket Stadium": "Visakhapatnam",
    "Himachal Pradesh Cricket Association Stadium": "Dharamsala",
}

# Each franchise's primary home ground, used to derive the `is_home` feature
# for seasons played at neutral venues (2009 South Africa, 2014 UAE, 2020 UAE,
# 2021 UAE leg), where the scheduled "home team" is not actually at home.
TEAM_HOME_VENUES: dict[str, str] = {
    "Chennai Super Kings": "MA Chidambaram Stadium",
    "Mumbai Indians": "Wankhede Stadium",
    "Royal Challengers Bengaluru": "M Chinnaswamy Stadium",
    "Kolkata Knight Riders": "Eden Gardens",
    "Sunrisers Hyderabad": "Rajiv Gandhi International Stadium",
    "Delhi Capitals": "Arun Jaitley Stadium",
    "Punjab Kings": "Punjab Cricket Association IS Bindra Stadium",
    "Rajasthan Royals": "Sawai Mansingh Stadium",
    "Gujarat Titans": "Narendra Modi Stadium",
    "Lucknow Super Giants": "Ekana Cricket Stadium",
    "Deccan Chargers": "Rajiv Gandhi International Stadium",
    "Pune Warriors": "Maharashtra Cricket Association Stadium",
    "Rising Pune Supergiant": "Maharashtra Cricket Association Stadium",
    "Gujarat Lions": "Saurashtra Cricket Association Stadium",
    "Kochi Tuskers Kerala": "Nehru Stadium",
}


# ---------------------------------------------------------------------------
# Match semantics
# ---------------------------------------------------------------------------
TOSS_DECISIONS = ("bat", "field")

# Playoff stage keywords found in feed round names / Cricsheet `match_number`.
PLAYOFF_KEYWORDS = ("final", "qualifier", "eliminator", "semi", "3rd place")

STANDARD_OVERS = 20
BALLS_PER_OVER = 6
POWERPLAY_OVERS = 6
DEATH_OVERS_FROM = 16  # Overs 16-20 are conventionally the "death" phase.

# Points awarded in the league stage - drives the standings table and the
# playoff-qualification Monte Carlo.
POINTS_WIN = 2
POINTS_TIE_OR_NR = 1
POINTS_LOSS = 0
PLAYOFF_SPOTS = 4


def team_color(team: str | None) -> str:
    """Return a franchise's brand colour, falling back to a neutral grey."""
    return TEAM_COLORS.get(team or "", DEFAULT_TEAM_COLOR)


def team_code(team: str | None) -> str:
    """Return a franchise's short code (``"CSK"``), or the name if unknown."""
    if not team:
        return ""
    return TEAM_CODES.get(team, team)
