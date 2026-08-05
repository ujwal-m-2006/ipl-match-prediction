-- ---------------------------------------------------------------------------
-- IPL Analytics warehouse - PostgreSQL schema
--
-- The application creates this schema automatically via SQLAlchemy
-- (`python scripts/init_db.py`). This file is the canonical reference for
-- DBAs, for review, and for provisioning a database out-of-band.
--
--   psql -U postgres -f sql/schema_postgres.sql
-- ---------------------------------------------------------------------------

CREATE DATABASE ipl
    WITH ENCODING 'UTF8'
         LC_COLLATE = 'en_US.UTF-8'
         LC_CTYPE = 'en_US.UTF-8'
         TEMPLATE = template0;

\connect ipl

-- ===========================================================================
-- Dimensions
-- ===========================================================================

-- A franchise, keyed on its canonical (current) name. Historical names are
-- folded into the current one during ingestion, so Delhi Daredevils results
-- accrue to Delhi Capitals.
CREATE TABLE teams (
    id            SERIAL PRIMARY KEY,
    name          VARCHAR(80)  NOT NULL UNIQUE,
    short_code    VARCHAR(10),
    is_active     BOOLEAN      NOT NULL DEFAULT FALSE,
    primary_color VARCHAR(16),
    logo_url      VARCHAR(400),
    created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- A cricket ground. Sponsor prefixes and trailing city names are stripped
-- during ingestion; the city is stored separately.
CREATE TABLE venues (
    id         SERIAL PRIMARY KEY,
    name       VARCHAR(160) NOT NULL UNIQUE,
    city       VARCHAR(80),
    country    VARCHAR(60)  DEFAULT 'India',
    created_at TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- A cricketer. `source_player_id` keeps the official feed's opaque GUID so a
-- later pull can match on ID rather than on a name spelling.
CREATE TABLE players (
    id               SERIAL PRIMARY KEY,
    name             VARCHAR(120) NOT NULL UNIQUE,
    source_player_id VARCHAR(80),
    short_name       VARCHAR(120),
    batting_style    VARCHAR(60),
    bowling_style    VARCHAR(80),
    role             VARCHAR(40),
    is_overseas      BOOLEAN,
    image_url        VARCHAR(400),
    created_at       TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ix_players_source_id ON players (source_player_id);

-- ===========================================================================
-- Facts
-- ===========================================================================

-- One fixture. Scheduled matches have NULL outcome columns and
-- is_completed = FALSE, which is exactly the filter the trainer uses.
CREATE TABLE matches (
    id                      SERIAL PRIMARY KEY,

    -- Identity & provenance. `match_key` namespaces the source's own ID,
    -- because the official feed and Cricsheet number matches independently.
    match_key               VARCHAR(60) NOT NULL UNIQUE,
    source                  VARCHAR(20) NOT NULL,
    source_match_id         VARCHAR(40),
    competition_id          INTEGER,

    -- Scheduling
    season                  INTEGER NOT NULL,
    match_date              DATE,
    start_datetime          TIMESTAMP,
    match_number            VARCHAR(40),
    stage                   VARCHAR(40),
    is_playoff              BOOLEAN NOT NULL DEFAULT FALSE,
    is_day_night            BOOLEAN,
    overs_per_innings       INTEGER DEFAULT 20,

    -- Location
    venue_id                INTEGER REFERENCES venues (id),
    city                    VARCHAR(80),
    is_neutral_venue        BOOLEAN NOT NULL DEFAULT FALSE,

    -- Participants
    team1_id                INTEGER REFERENCES teams (id),
    team2_id                INTEGER REFERENCES teams (id),
    home_team_id            INTEGER REFERENCES teams (id),
    away_team_id            INTEGER REFERENCES teams (id),

    -- Toss
    toss_winner_id          INTEGER REFERENCES teams (id),
    toss_decision           VARCHAR(10),

    -- Innings order
    first_batting_team_id   INTEGER REFERENCES teams (id),
    second_batting_team_id  INTEGER REFERENCES teams (id),

    -- Outcome
    is_completed            BOOLEAN NOT NULL DEFAULT FALSE,
    winner_id               INTEGER REFERENCES teams (id),
    result_type             VARCHAR(20),
    win_margin_runs         INTEGER,
    win_margin_wickets      INTEGER,
    is_tie                  BOOLEAN NOT NULL DEFAULT FALSE,
    is_no_result            BOOLEAN NOT NULL DEFAULT FALSE,
    is_super_over           BOOLEAN NOT NULL DEFAULT FALSE,
    is_dls_applied          BOOLEAN NOT NULL DEFAULT FALSE,
    target_runs             INTEGER,
    player_of_match_id      INTEGER REFERENCES players (id),
    result_summary          VARCHAR(300),

    -- Officials
    umpire1                 VARCHAR(80),
    umpire2                 VARCHAR(80),
    third_umpire            VARCHAR(80),
    match_referee           VARCHAR(80),

    ingested_at             TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT ck_matches_toss_decision
        CHECK (toss_decision IS NULL OR toss_decision IN ('bat', 'field')),
    CONSTRAINT ck_matches_teams_differ
        CHECK (team1_id IS NULL OR team2_id IS NULL OR team1_id <> team2_id)
);

CREATE INDEX ix_matches_season       ON matches (season);
CREATE INDEX ix_matches_date         ON matches (match_date);
CREATE INDEX ix_matches_completed    ON matches (is_completed);
CREATE INDEX ix_matches_season_date  ON matches (season, match_date);
CREATE INDEX ix_matches_teams        ON matches (team1_id, team2_id);
CREATE INDEX ix_matches_winner       ON matches (winner_id);
CREATE INDEX ix_matches_venue        ON matches (venue_id);

-- Per-innings totals, with phase splits derived from ball-by-ball data.
CREATE TABLE innings (
    id                 SERIAL PRIMARY KEY,
    match_id           INTEGER NOT NULL REFERENCES matches (id) ON DELETE CASCADE,
    innings_no         INTEGER NOT NULL,
    batting_team_id    INTEGER REFERENCES teams (id),
    bowling_team_id    INTEGER REFERENCES teams (id),

    runs               INTEGER,
    wickets            INTEGER,
    overs              REAL,      -- cricket notation, e.g. 19.4
    balls              INTEGER,   -- legal deliveries bowled
    run_rate           REAL,

    extras             INTEGER,
    byes               INTEGER,
    leg_byes           INTEGER,
    wides              INTEGER,
    no_balls           INTEGER,
    penalty            INTEGER,

    powerplay_runs     INTEGER,
    powerplay_wickets  INTEGER,
    middle_runs        INTEGER,
    middle_wickets     INTEGER,
    death_runs         INTEGER,
    death_wickets      INTEGER,

    fours              INTEGER,
    sixes              INTEGER,
    dot_balls          INTEGER,
    target             INTEGER,
    is_declared        BOOLEAN NOT NULL DEFAULT FALSE,

    CONSTRAINT uq_innings_match_no UNIQUE (match_id, innings_no),
    CONSTRAINT ck_innings_wickets  CHECK (wickets IS NULL OR wickets BETWEEN 0 AND 10)
);

CREATE INDEX ix_innings_match ON innings (match_id);

CREATE TABLE batting_cards (
    id                SERIAL PRIMARY KEY,
    match_id          INTEGER NOT NULL REFERENCES matches (id) ON DELETE CASCADE,
    innings_no        INTEGER NOT NULL,
    team_id           INTEGER REFERENCES teams (id),
    player_id         INTEGER NOT NULL REFERENCES players (id),

    batting_position  INTEGER,
    runs              INTEGER DEFAULT 0,
    balls             INTEGER DEFAULT 0,
    fours             INTEGER DEFAULT 0,
    sixes             INTEGER DEFAULT 0,
    dot_balls         INTEGER,
    strike_rate       REAL,

    is_out            BOOLEAN NOT NULL DEFAULT FALSE,
    dismissal_kind    VARCHAR(40),
    dismissal_text    VARCHAR(200),
    bowler_id         INTEGER REFERENCES players (id),
    fielder_id        INTEGER REFERENCES players (id),
    wicket_number     INTEGER,

    CONSTRAINT uq_batting_card UNIQUE (match_id, innings_no, player_id)
);

CREATE INDEX ix_batting_match  ON batting_cards (match_id);
CREATE INDEX ix_batting_player ON batting_cards (player_id);

CREATE TABLE bowling_cards (
    id             SERIAL PRIMARY KEY,
    match_id       INTEGER NOT NULL REFERENCES matches (id) ON DELETE CASCADE,
    innings_no     INTEGER NOT NULL,
    team_id        INTEGER REFERENCES teams (id),
    player_id      INTEGER NOT NULL REFERENCES players (id),

    bowling_order  INTEGER,
    overs          REAL DEFAULT 0,
    balls          INTEGER,
    maidens        INTEGER DEFAULT 0,
    runs_conceded  INTEGER DEFAULT 0,
    wickets        INTEGER DEFAULT 0,
    wides          INTEGER DEFAULT 0,
    no_balls       INTEGER DEFAULT 0,
    dot_balls      INTEGER,
    economy        REAL,

    CONSTRAINT uq_bowling_card UNIQUE (match_id, innings_no, player_id)
);

CREATE INDEX ix_bowling_match  ON bowling_cards (match_id);
CREATE INDEX ix_bowling_player ON bowling_cards (player_id);

CREATE TABLE fall_of_wickets (
    id          SERIAL PRIMARY KEY,
    match_id    INTEGER NOT NULL REFERENCES matches (id) ON DELETE CASCADE,
    innings_no  INTEGER NOT NULL,
    wicket_no   INTEGER NOT NULL,
    player_id   INTEGER REFERENCES players (id),
    team_id     INTEGER REFERENCES teams (id),
    fall_score  INTEGER,
    fall_overs  REAL,

    CONSTRAINT uq_fow UNIQUE (match_id, innings_no, wicket_no)
);

CREATE TABLE partnerships (
    id                 SERIAL PRIMARY KEY,
    match_id           INTEGER NOT NULL REFERENCES matches (id) ON DELETE CASCADE,
    innings_no         INTEGER NOT NULL,
    wicket_no          INTEGER NOT NULL,
    team_id            INTEGER REFERENCES teams (id),

    striker_id         INTEGER REFERENCES players (id),
    non_striker_id     INTEGER REFERENCES players (id),
    runs               INTEGER,
    balls              INTEGER,
    striker_runs       INTEGER,
    striker_balls      INTEGER,
    non_striker_runs   INTEGER,
    non_striker_balls  INTEGER,
    extras             INTEGER,
    start_over         REAL,
    end_over           REAL,
    is_unbroken        BOOLEAN NOT NULL DEFAULT FALSE,

    CONSTRAINT uq_partnership UNIQUE (match_id, innings_no, wicket_no)
);

-- The ball-by-ball grain: ~270k rows for 2008-2026. Powers the in-play chase
-- model, phase analytics and player strike/economy profiles.
CREATE TABLE deliveries (
    id                   SERIAL PRIMARY KEY,
    match_id             INTEGER NOT NULL REFERENCES matches (id) ON DELETE CASCADE,
    innings_no           INTEGER NOT NULL,
    over_no              INTEGER NOT NULL,   -- 1-based
    ball_no              INTEGER NOT NULL,   -- position within the over
    ball_seq             INTEGER NOT NULL,   -- 1-based within the innings

    batting_team_id      INTEGER REFERENCES teams (id),
    bowling_team_id      INTEGER REFERENCES teams (id),
    batter_id            INTEGER REFERENCES players (id),
    non_striker_id       INTEGER REFERENCES players (id),
    bowler_id            INTEGER REFERENCES players (id),

    batter_runs          INTEGER DEFAULT 0,
    extra_runs           INTEGER DEFAULT 0,
    total_runs           INTEGER DEFAULT 0,

    is_wide              BOOLEAN NOT NULL DEFAULT FALSE,
    is_no_ball           BOOLEAN NOT NULL DEFAULT FALSE,
    is_bye               BOOLEAN NOT NULL DEFAULT FALSE,
    is_leg_bye           BOOLEAN NOT NULL DEFAULT FALSE,
    is_legal             BOOLEAN NOT NULL DEFAULT TRUE,
    is_four              BOOLEAN NOT NULL DEFAULT FALSE,
    is_six               BOOLEAN NOT NULL DEFAULT FALSE,
    is_wicket            BOOLEAN NOT NULL DEFAULT FALSE,
    wicket_type          VARCHAR(40),
    dismissed_player_id  INTEGER REFERENCES players (id),

    -- Running state after this ball, precomputed so live inference needs no
    -- window function.
    cumulative_runs      INTEGER,
    cumulative_wickets   INTEGER,

    CONSTRAINT uq_delivery UNIQUE (match_id, innings_no, ball_seq)
);

CREATE INDEX ix_deliveries_match         ON deliveries (match_id);
CREATE INDEX ix_deliveries_match_innings ON deliveries (match_id, innings_no);
CREATE INDEX ix_deliveries_batter        ON deliveries (batter_id);
CREATE INDEX ix_deliveries_bowler        ON deliveries (bowler_id);

CREATE TABLE match_players (
    id               SERIAL PRIMARY KEY,
    match_id         INTEGER NOT NULL REFERENCES matches (id) ON DELETE CASCADE,
    team_id          INTEGER NOT NULL REFERENCES teams (id),
    player_id        INTEGER NOT NULL REFERENCES players (id),

    is_playing_xi    BOOLEAN NOT NULL DEFAULT TRUE,
    is_captain       BOOLEAN NOT NULL DEFAULT FALSE,
    is_wicketkeeper  BOOLEAN NOT NULL DEFAULT FALSE,
    is_overseas      BOOLEAN,
    is_impact_sub    BOOLEAN NOT NULL DEFAULT FALSE,
    playing_order    INTEGER,
    role             VARCHAR(40),

    CONSTRAINT uq_match_player UNIQUE (match_id, team_id, player_id)
);

CREATE INDEX ix_match_players_match  ON match_players (match_id);
CREATE INDEX ix_match_players_player ON match_players (player_id);

-- ===========================================================================
-- Operations
-- ===========================================================================

-- Audit trail for the data pipeline. The dashboard's Admin page reads the
-- latest row to report data freshness.
CREATE TABLE ingestion_runs (
    id                   SERIAL PRIMARY KEY,
    started_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at          TIMESTAMP,
    status               VARCHAR(20) NOT NULL DEFAULT 'running',
    trigger              VARCHAR(30),
    sources              VARCHAR(120),
    seasons              VARCHAR(200),

    matches_seen         INTEGER DEFAULT 0,
    matches_inserted     INTEGER DEFAULT 0,
    matches_updated      INTEGER DEFAULT 0,
    matches_skipped      INTEGER DEFAULT 0,
    deliveries_inserted  INTEGER DEFAULT 0,
    duration_seconds     REAL,
    message              TEXT
);

-- ===========================================================================
-- Convenience views
-- ===========================================================================

-- Matches with human-readable names, mirroring the join the application uses.
CREATE VIEW v_matches AS
SELECT
    m.id            AS match_id,
    m.match_key,
    m.season,
    m.match_date,
    m.stage,
    m.is_playoff,
    v.name          AS venue,
    m.city,
    t1.name         AS team1,
    t2.name         AS team2,
    tt.name         AS toss_winner,
    m.toss_decision,
    tf.name         AS first_batting_team,
    tw.name         AS winner,
    m.result_type,
    m.win_margin_runs,
    m.win_margin_wickets,
    m.result_summary,
    p.name          AS player_of_match,
    m.is_completed
FROM matches m
LEFT JOIN venues  v  ON v.id  = m.venue_id
LEFT JOIN teams   t1 ON t1.id = m.team1_id
LEFT JOIN teams   t2 ON t2.id = m.team2_id
LEFT JOIN teams   tt ON tt.id = m.toss_winner_id
LEFT JOIN teams   tf ON tf.id = m.first_batting_team_id
LEFT JOIN teams   tw ON tw.id = m.winner_id
LEFT JOIN players p  ON p.id  = m.player_of_match_id;

-- All-time franchise record, computed from both team slots.
CREATE VIEW v_team_records AS
WITH sides AS (
    SELECT team1_id AS team_id, winner_id FROM matches WHERE is_completed
    UNION ALL
    SELECT team2_id AS team_id, winner_id FROM matches WHERE is_completed
)
SELECT
    t.name                                            AS team,
    COUNT(*)                                          AS matches,
    SUM(CASE WHEN s.winner_id = s.team_id THEN 1 ELSE 0 END) AS wins,
    ROUND(
        100.0 * SUM(CASE WHEN s.winner_id = s.team_id THEN 1 ELSE 0 END) / COUNT(*),
        2
    )                                                 AS win_pct
FROM sides s
JOIN teams t ON t.id = s.team_id
GROUP BY t.name
ORDER BY win_pct DESC;
